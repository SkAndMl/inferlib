import hashlib
import threading
import torch

from collections import Counter, deque
from dataclasses import dataclass
from torch import Tensor
from typing import Optional

from inferlib.core.engine.sequence import Sequence


@dataclass
class _PagePool:
    num_pages: int
    num_layers: int
    num_heads: int
    page_size: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        size = (
            self.num_pages,
            self.num_layers,
            self.num_heads,
            self.page_size,
            self.head_dim,
        )
        self._key_pool = torch.empty(size=size, dtype=self.dtype, device=self.device)
        self._value_pool = torch.empty(size=size, dtype=self.dtype, device=self.device)

    def write(
        self,
        page_ids: Tensor,
        layer_id: int,
        kv: tuple[Tensor, Tensor],
        offsets: Optional[Tensor] = None,
    ):
        assert layer_id < self.num_layers

        k, v = kv
        k = k.to(device=self.device, dtype=self.dtype, copy=False)
        v = v.to(device=self.device, dtype=self.dtype, copy=False)

        if offsets is not None:
            assert (offsets < self.page_size).all()
            assert k.shape == v.shape == (len(page_ids), self.num_heads, 1, self.head_dim)

            self._key_pool[page_ids, layer_id, :, offsets, :] = k.squeeze(dim=2)
            self._value_pool[page_ids, layer_id, :, offsets, :] = v.squeeze(dim=2)

        else:
            length = k.shape[2]
            self._key_pool[page_ids, layer_id, :, :length, :] = k
            self._value_pool[page_ids, layer_id, :, :length, :] = v

    def read(self, page_ids: Tensor, layer_id: int) -> tuple[Tensor, Tensor]:
        assert layer_id < self.num_layers
        k = self._key_pool[page_ids, layer_id, ...]
        v = self._value_pool[page_ids, layer_id, ...]
        return k, v


@dataclass
class Page:
    page_id: int
    page_hash: str = None
    parent_hash: Optional[str] = None
    refcount: int = 0


@dataclass(frozen=True)
class _PrefillPlan:
    cached_page_ids: list[int]
    fresh_pages_needed: int
    prefix_cached_tokens: int


@dataclass
class PageManager:
    num_pages: int
    num_layers: int
    num_heads: int
    page_size: int
    head_dim: int
    max_pages_per_sequence: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        self._page_pool = _PagePool(
            num_pages=self.num_pages,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            page_size=self.page_size,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._page_table: dict[int, list[int]] = {}

        self._free_pages = deque(range(self.num_pages))
        self._page_info: list[Page] = [Page(page_id=i) for i in range(self.num_pages)]

        self._cache: dict[str, int] = {}  # hash -> page_id

        self._state_lock = threading.Lock()

    def _matched_prefix(self, sequence: Sequence) -> list[int]:
        parent_hash = None
        matched_page_ids = []
        for i in range(0, len(sequence), self.page_size):
            page_tokens = sequence.tokens[i : i + self.page_size]
            if len(page_tokens) != self.page_size:
                break

            page_hash = self._stable_hash((parent_hash, page_tokens))
            if page_hash not in self._cache:
                break

            matched_page_ids.append(self._cache[page_hash])
            parent_hash = page_hash

        return matched_page_ids

    def plan_prefill(self, sequence: Sequence, n: int) -> Optional[_PrefillPlan]:
        with self._state_lock:
            matched_prefix = self._matched_prefix(sequence)
            fresh_pages_needed = n - len(matched_prefix)
            if fresh_pages_needed == 0:
                # cannot cache the whole prompt
                # since we need atleast one uncached page for decoding
                fresh_pages_needed = 1
                matched_prefix = matched_prefix[:-1]

            if len(self._free_pages) < fresh_pages_needed:
                return None

            return _PrefillPlan(
                cached_page_ids=matched_prefix[:],
                fresh_pages_needed=fresh_pages_needed,
                prefix_cached_tokens=len(matched_prefix) * self.page_size,
            )

    def allocate_prefill(self, sequence: Sequence, plan: _PrefillPlan) -> None:
        with self._state_lock:
            assert sequence.s_id not in self._page_table, (
                f"Prefill sequence {sequence.s_id} already has committed pages"
            )
            assert len(self._free_pages) >= plan.fresh_pages_needed

            fresh_page_ids = [self._free_pages.popleft() for _ in range(plan.fresh_pages_needed)]
            self._retain_pages(plan.cached_page_ids)
            self._retain_pages(fresh_page_ids)

            self._page_table[sequence.s_id] = plan.cached_page_ids + fresh_page_ids
            sequence.prefix_cached_tokens = plan.prefix_cached_tokens

            if __debug__:
                self._check_invariants()

    def append_decode_pages(self, sequence: Sequence, n: int) -> bool:
        with self._state_lock:
            if n == 0:
                return True
            if len(self._free_pages) < n:
                return False

            page_ids = [self._free_pages.popleft() for _ in range(n)]
            self._retain_pages(page_ids)
            self._page_table[sequence.s_id].extend(page_ids)

            if __debug__:
                self._check_invariants()

            return True

    def free(self, sequence: Sequence):
        with self._state_lock:
            page_ids: list[int] = self._page_table.pop(sequence.s_id)
            for page_id in page_ids:
                self._release_page(page_id, sequence)

            if __debug__:
                self._check_invariants()

    def evict(self, sequence: Sequence):
        with self._state_lock:
            pid = self._page_table[sequence.s_id].pop(0)  # TODO: O(n) op, optimize later
            self._release_page(pid, sequence)

            if __debug__:
                self._check_invariants()

    def prefill(
        self,
        sequences: list[Sequence],
        layer_id: int,
        kv: tuple[Tensor, Tensor],
        cached_pages_offset: int = 0,
    ):
        """Write KV computed during prefill into the page pool.

        When ``cached_pages_offset > 0`` the first *offset* pages already
        contain valid KV (from a previous prefix-cache hit) and are skipped.
        The *kv* tensors only contain data for the **uncached** tokens, so
        page ``cached_pages_offset + j`` reads from ``kv[:, :, j*page_size : (j+1)*page_size, :]``.

        The hash chain is maintained for every full page (including those
        whose parent is a cached page) so that future requests can match
        longer prefixes.
        """
        k, v = kv
        num_pages = len(self._page_table[sequences[0].s_id])
        assert all(len(self._page_table[seq.s_id]) == num_pages for seq in sequences)

        for i in range(cached_pages_offset, num_pages):
            page_ids = torch.tensor(
                [self._page_table[seq.s_id][i] for seq in sequences],
                dtype=torch.long,
                device=self.device,
            )

            # kv tensors are indexed from 0 — they only cover uncached tokens
            kv_idx = i - cached_pages_offset
            _k = k[:, :, kv_idx * self.page_size : (kv_idx + 1) * self.page_size, :]
            _v = v[:, :, kv_idx * self.page_size : (kv_idx + 1) * self.page_size, :]
            self._page_pool.write(page_ids=page_ids, layer_id=layer_id, kv=(_k, _v))

            ## prefix cache — hash chain uses the FULL prompt position
            if layer_id != self.num_layers - 1:
                continue
            for sequence in sequences:
                page_tokens: list[int] = sequence.prompt_tokens[i * self.page_size : (i + 1) * self.page_size]
                if len(page_tokens) != self.page_size:  # partially filled page; skip
                    continue

                page_id = self._page_table[sequence.s_id][i]
                parent_hash = None
                if i > 0:
                    parent_page_id = self._page_table[sequence.s_id][i - 1]
                    parent_hash = self._page_info[parent_page_id].page_hash

                page_hash = self._stable_hash((parent_hash, page_tokens))
                self._page_info[page_id].parent_hash = parent_hash
                self._page_info[page_id].page_hash = page_hash

                self._cache[page_hash] = page_id
                sequence.cached_pages += 1

    def write(
        self,
        sequences: list[Sequence],
        layer_id: int,
        kv: tuple[Tensor, Tensor],
    ):
        page_ids = torch.tensor(
            [self._page_table[sequence.s_id][-1] for sequence in sequences],
            device=self.device,
            dtype=torch.long,
        )
        offsets = torch.tensor(
            data=[(len(sequence) - sequence.tokens_evicted - 1) % self.page_size for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        self._page_pool.write(
            page_ids=page_ids,
            layer_id=layer_id,
            offsets=offsets,
            kv=kv,
        )

        ## cache
        if layer_id != self.num_layers - 1:
            return

        for sequence in sequences:
            # check if last page of the sequence has been filled
            if (len(sequence) - sequence.tokens_evicted) % self.page_size != 0:
                continue
            page_ids = self._page_table[sequence.s_id]
            page_id = page_ids[-1]

            parent_hash = None
            if len(page_ids) > 1:
                parent_hash = self._page_info[page_ids[-2]].page_hash

            page_tokens = sequence.tokens[-self.page_size :]
            page_hash = self._stable_hash((parent_hash, page_tokens))
            self._page_info[page_id].parent_hash = parent_hash
            self._page_info[page_id].page_hash = page_hash

            self._cache[page_hash] = page_id
            sequence.cached_pages += 1

    def read_page(self, page_id: int, layer_id: int):
        page_ids = torch.tensor(data=[page_id], device=self.device)
        k, v = self._page_pool.read(page_ids=page_ids, layer_id=layer_id)
        return k, v

    def read_from_page_table(
        self,
        page_table: Tensor,
        layer_id: int,
        page_idx: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        page_ids = page_table[:, page_idx]
        valid = page_ids != -1
        safe_page_ids = page_ids.masked_fill(~valid, 0)

        k, v = self._page_pool.read(page_ids=safe_page_ids, layer_id=layer_id)
        valid = valid[:, None, None, None]
        k = k * valid
        v = v * valid
        return k, v, valid

    def build_page_table(self, sequences: list[Sequence]) -> Tensor:
        page_ids: list[list[int]] = [self._page_table.get(sequence.s_id, []) for sequence in sequences]
        max_pages = max(len(_) for _ in page_ids)
        page_table = torch.full(size=(len(sequences), max_pages), fill_value=-1, dtype=torch.long)
        for i, _page_ids in enumerate(page_ids):
            page_table[i, : len(_page_ids)] = torch.tensor(_page_ids, dtype=torch.long)

        return page_table.to(device=self.device)

    def get_num_pages(self, sequence: Sequence) -> int:
        return len(self._page_table.get(sequence.s_id, []))

    def get_cached_page_ids(self, sequence: Sequence) -> list[int]:
        return self._page_table.get(sequence.s_id, [])[: sequence.cached_pages]

    @property
    def num_free_pages(self) -> int:
        with self._state_lock:
            return len(self._free_pages)

    def _stable_hash(self, parts: tuple[object, ...]) -> str:
        h = hashlib.sha256()
        for part in parts:
            h.update(repr(part).encode("utf-8"))
            h.update(b"|")
        return h.hexdigest()

    def _retain_pages(self, page_ids: list[int]) -> None:
        for page_id in page_ids:
            self._page_info[page_id].refcount += 1

    def _release_page(self, page_id: int, sequence: Sequence) -> None:
        info = self._page_info[page_id]
        info.refcount -= 1
        assert info.refcount >= 0
        if info.refcount > 0:
            return

        if info.page_hash is not None:
            if self._cache.get(info.page_hash) == page_id:
                del self._cache[info.page_hash]
            sequence.cached_pages -= 1
            assert sequence.cached_pages >= 0
        info.page_hash = None
        info.parent_hash = None
        self._free_pages.append(page_id)

    def _check_invariants(self):
        allocated = [p for s_id in self._page_table for p in self._page_table[s_id]]

        free_set = set(self._free_pages)
        allocated_set = set(allocated)

        allocated_counts = Counter(allocated)

        assert len(free_set) == len(self._free_pages), "Duplicate page_id in _free_pages"

        assert free_set.isdisjoint(allocated_set), "Page appears in both free and allocated set"

        all_pages = set(range(self.num_pages))
        union = free_set | allocated_set
        assert union.issubset(all_pages), "Invalid page id found"
        assert len(union) == self.num_pages, f"Leak/dup: accounted={len(union)}, expected={self.num_pages}"

        for p in range(self.num_pages):
            expected = allocated_counts.get(p, 0)
            actual = self._page_info[p].refcount
            assert expected == actual, f"refcount mismatch page={p}: {actual} != {expected}"


__all__ = ["PageManager"]
