/**
 * 모든 마크다운 파일에 published: true 필드 추가
 * published 필드가 없는 파일만 업데이트
 */

const fs = require('fs');
const path = require('path');

const contentsDir = path.join(__dirname, '../contents');

// 재귀적으로 모든 .md 파일 찾기
function findMarkdownFiles(dir) {
  let results = [];
  const list = fs.readdirSync(dir);

  list.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      results = results.concat(findMarkdownFiles(filePath));
    } else if (file.endsWith('.md')) {
      results.push(filePath);
    }
  });

  return results;
}

// 모든 마크다운 파일 찾기
const files = findMarkdownFiles(contentsDir);

let updatedCount = 0;
let skippedCount = 0;

files.forEach((filePath) => {
  const content = fs.readFileSync(filePath, 'utf-8');

  // published 필드가 이미 있는지 확인
  if (/^published:/m.test(content)) {
    console.log(`⏭️  Skipped (already has published): ${path.relative(contentsDir, filePath)}`);
    skippedCount++;
    return;
  }

  // frontmatter 끝 (---) 찾기
  const lines = content.split('\n');
  const frontmatterEndIndex = lines.findIndex((line, index) => index > 0 && line.trim() === '---');

  if (frontmatterEndIndex === -1) {
    console.log(`❌ No frontmatter found: ${path.relative(contentsDir, filePath)}`);
    return;
  }

  // published: true를 frontmatter 끝 직전에 추가
  lines.splice(frontmatterEndIndex, 0, 'published: true');

  const newContent = lines.join('\n');
  fs.writeFileSync(filePath, newContent, 'utf-8');

  console.log(`✅ Updated: ${path.relative(contentsDir, filePath)}`);
  updatedCount++;
});

console.log('\n=== Summary ===');
console.log(`✅ Updated: ${updatedCount} files`);
console.log(`⏭️  Skipped: ${skippedCount} files`);
console.log(`📁 Total: ${files.length} files`);
