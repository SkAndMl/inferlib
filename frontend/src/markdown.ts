import DOMPurify from "dompurify";
import katex from "katex";
import { marked } from "marked";

marked.setOptions({
  gfm: true,
  breaks: true,
});

interface MathPlaceholderResult {
  text: string;
  mathHtml: string[];
}

export interface ContentSegment {
  type: "text" | "thinking";
  content: string;
  inProgress?: boolean;
}

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderKatexExpression(expression: string, displayMode: boolean): string | null {
  try {
    return katex.renderToString(expression, {
      displayMode,
      throwOnError: false,
    });
  } catch {
    return null;
  }
}

function applyMathPlaceholders(markdown: string): MathPlaceholderResult {
  let text = markdown;
  const mathHtml: string[] = [];
  const codeBlocks: string[] = [];

  text = text.replace(/```[\s\S]*?```/g, (block) => {
    const token = `@@MATH_CODE_${codeBlocks.length}@@`;
    codeBlocks.push(block);
    return token;
  });

  text = text.replace(/`[^`\n]*`/g, (span) => {
    const token = `@@MATH_CODE_${codeBlocks.length}@@`;
    codeBlocks.push(span);
    return token;
  });

  function pushMath(expression: string, displayMode: boolean): string {
    const token = `@@MATH_${mathHtml.length}@@`;
    const rendered = renderKatexExpression(String(expression).trim(), displayMode);
    if (rendered) {
      mathHtml.push(rendered);
    } else {
      const wrapped = displayMode ? `$$${expression}$$` : `$${expression}$`;
      mathHtml.push(escapeHtml(wrapped));
    }
    return token;
  }

  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_match, expression) =>
    pushMath(expression, true)
  );

  text = text.replace(/(?<!\\)\$([^\n$]+?)(?<!\\)\$/g, (_match, expression) =>
    pushMath(expression, false)
  );

  text = text.replace(/@@MATH_CODE_(\d+)@@/g, (_match, index) => {
    return codeBlocks[Number(index)] || "";
  });

  return { text, mathHtml };
}

function restoreMathPlaceholders(html: string, mathHtml: string[]): string {
  return html.replace(/@@MATH_(\d+)@@/g, (_match, index) => mathHtml[Number(index)] || "");
}

export function renderMarkdown(markdown: string): string {
  const source = String(markdown ?? "");
  const { text, mathHtml } = applyMathPlaceholders(source);
  const rawHtml = marked.parse(text, { async: false }) as string;
  const cleanHtml = DOMPurify.sanitize(rawHtml);
  return restoreMathPlaceholders(cleanHtml, mathHtml);
}

export function splitThinkingSegments(content: string): ContentSegment[] {
  const source = String(content ?? "");
  const segments: ContentSegment[] = [];
  let cursor = 0;
  const openTag = "<think>";
  const closeTag = "</think>";

  while (cursor < source.length) {
    const start = source.indexOf(openTag, cursor);
    if (start === -1) {
      segments.push({
        type: "text",
        content: source.slice(cursor),
      });
      break;
    }

    if (start > cursor) {
      segments.push({
        type: "text",
        content: source.slice(cursor, start),
      });
    }

    const thinkingStart = start + openTag.length;
    const end = source.indexOf(closeTag, thinkingStart);

    if (end === -1) {
      segments.push({
        type: "thinking",
        content: source.slice(thinkingStart),
        inProgress: true,
      });
      cursor = source.length;
      break;
    }

    segments.push({
      type: "thinking",
      content: source.slice(thinkingStart, end),
      inProgress: false,
    });
    cursor = end + closeTag.length;
  }

  if (!segments.length) {
    return [{ type: "text", content: source }];
  }

  return segments;
}
