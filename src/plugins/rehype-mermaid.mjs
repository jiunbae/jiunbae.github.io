import { visit } from 'unist-util-visit';

export function rehypeMermaid() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      // Look for pre > code.language-mermaid
      if (
        node.tagName === 'pre' &&
        node.children?.[0]?.tagName === 'code' &&
        node.children[0].properties?.className?.includes('language-mermaid')
      ) {
        const codeNode = node.children[0];
        const mermaidCode = codeNode.children?.[0]?.value || '';

        // Replace with a div.mermaid
        parent.children[index] = {
          type: 'element',
          tagName: 'div',
          properties: { className: ['mermaid'] },
          children: [{ type: 'text', value: mermaidCode }],
        };
      }
    });
  };
}

export default rehypeMermaid;
