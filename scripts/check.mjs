#!/usr/bin/env node
/**
 * VoxCPM2-C Project Verification Script
 * Controlled Workflow V4.1 JS/TS Fastloop
 *
 * Verifies project structure, file existence, and basic content integrity.
 * Run with: node scripts/check.mjs
 */

import { existsSync, readFileSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

let passed = 0;
let failed = 0;
let warnings = 0;

function check(condition, label, detail = '') {
  if (condition) {
    console.log(`  ✅ ${label}`);
    passed++;
  } else {
    console.log(`  ❌ ${label} ${detail}`);
    failed++;
  }
}

function warn(label, detail = '') {
  console.log(`  ⚠️  ${label} ${detail}`);
  warnings++;
}

function checkFile(path, label) {
  const fullPath = join(ROOT, path);
  const exists = existsSync(fullPath) && statSync(fullPath).isFile();
  check(exists, `${label} (${path})`, exists ? '' : '— MISSING');
  return exists;
}

function checkDir(path, label) {
  const fullPath = join(ROOT, path);
  const exists = existsSync(fullPath) && statSync(fullPath).isDirectory();
  check(exists, `${label} (${path}/)`, exists ? '' : '— MISSING');
  return exists;
}

function checkFileSize(path, minBytes) {
  const fullPath = join(ROOT, path);
  if (!existsSync(fullPath)) return 0;
  const size = statSync(fullPath).size;
  check(size >= minBytes, `${path} size >= ${minBytes} bytes`, `(actual: ${size})`);
  return size;
}

function checkIncludesHeader(path, searchStr) {
  const fullPath = join(ROOT, path);
  if (!existsSync(fullPath)) return;
  const content = readFileSync(fullPath, 'utf-8');
  check(content.includes(searchStr), `${path} includes "${searchStr}"`);
}

console.log('══════════════════════════════════════════════');
console.log(' VoxCPM2-C — Project Structure Verification');
console.log('══════════════════════════════════════════════\n');

// ─── Phase 0 Checklist ───
console.log('📁 Phase 0: Build System');
checkFile('CMakeLists.txt', 'CMake build system');
checkFile('Makefile', 'GNU Make backup');
checkFile('CMakeLists.txt', 'CMakeLists.txt exists');
checkIncludesHeader('CMakeLists.txt', 'project(voxcpm2-c');

console.log('\n📁 Phase 0: Public API Header');
checkFile('include/voxcpm.h', 'Public C API header');
checkIncludesHeader('include/voxcpm.h', 'voxcpm_create');
checkIncludesHeader('include/voxcpm.h', 'voxcpm_generate');
checkIncludesHeader('include/voxcpm.h', 'voxcpm_free');

console.log('\n📁 Phase 0: Tensor Library');
checkFile('include/tensor.h', 'Tensor header');
checkFile('src/tensor.c', 'Tensor implementation');
checkIncludesHeader('include/tensor.h', 'tensor_create');
checkIncludesHeader('include/tensor.h', 'tensor_matmul');
checkIncludesHeader('include/tensor.h', 'tensor_softmax');
checkIncludesHeader('include/tensor.h', 'tensor_rms_norm');
checkIncludesHeader('include/tensor.h', 'tensor_rotary_emb');
checkIncludesHeader('include/tensor.h', 'tensor_silu');

console.log('\n📁 Phase 0: NN Layer Headers');
checkFile('include/nn.h', 'NN header');
checkIncludesHeader('include/nn.h', 'TransformerBlock');
checkIncludesHeader('include/nn.h', 'SwiGLU');
checkIncludesHeader('include/nn.h', 'Attention');

console.log('\n📁 Phase 0: Platform Abstraction');
checkFile('include/platform.h', 'Platform header');
checkFile('src/platform.c', 'Platform implementation');
checkIncludesHeader('include/platform.h', 'MmapFile');
checkIncludesHeader('include/platform.h', 'thread_pool_create');
checkIncludesHeader('include/platform.h', 'Timer');

console.log('\n📁 Phase 0: Model Structure');
checkFile('include/model.h', 'Model header');
checkIncludesHeader('include/model.h', 'VoxCPMConfig');
checkIncludesHeader('include/model.h', 'VoxCPMModel');

console.log('\n📁 Phase 0: Source Files');
const sourceFiles = [
  'src/tensor.c', 'src/nn.c', 'src/model.c',
  'src/loc_enc.c', 'src/tslm.c', 'src/ralm.c',
  'src/loc_dit.c', 'src/audio_vae.c', 'src/audio.c',
  'src/tokenizer.c', 'src/sampler.c', 'src/utils.c',
  'src/platform.c', 'src/main.c'
];
for (const f of sourceFiles) {
  checkFile(f, `Source: ${f}`);
}

console.log('\n📁 Phase 0: Additional Headers');
const headerFiles = [
  'include/audio.h', 'include/sampler.h',
  'include/tensor.h', 'include/nn.h',
  'include/model.h', 'include/platform.h',
  'include/voxcpm.h'
];
for (const f of headerFiles) {
  checkFile(f, `Header: ${f}`);
}

console.log('\n📁 Phase 0: Tests');
checkFile('tests/test_tensor.c', 'Tensor tests');

console.log('\n📁 Phase 0: Examples & Benchmarks');
checkFile('examples/tts_minimal.c', 'TTS minimal example');
checkFile('benchmarks/bench_rtf.c', 'RTF benchmark stub');

console.log('\n📁 Project Documentation');
checkFile('spec.md', 'Specification document');
checkFile('plan.md', 'Implementation plan');
checkFile('todos.md', 'Task list');
checkFile('test.md', 'Test plan');
checkFile('final.md', 'Final delivery doc');
checkFile('AGENTS.md', 'Agent workflow guide');
checkFile('TEAM-C.md', 'C team conventions');

console.log('\n📁 Source File Sizes (>0 bytes)');
const allSources = [
  ...sourceFiles,
  ...headerFiles,
  'tests/test_tensor.c',
  'examples/tts_minimal.c',
  'benchmarks/bench_rtf.c',
  'CMakeLists.txt',
  'Makefile'
];
for (const f of allSources) {
  checkFileSize(f, 1);
}

console.log('\n📁 Key Source Size Checks');
// Tensor.c should be substantial since it has all the core implementations
const tensorSize = checkFileSize('src/tensor.c', 1000);
if (tensorSize > 0) {
  const lineCount = readFileSync(join(ROOT, 'src/tensor.c'), 'utf-8').split('\n').length;
  check(lineCount > 100, `tensor.c is substantial (${lineCount} lines)`);
}

const modelSize = checkFileSize('src/model.c', 500);
if (modelSize > 0) {
  const lineCount = readFileSync(join(ROOT, 'src/model.c'), 'utf-8').split('\n').length;
  check(lineCount > 50, `model.c is substantial (${lineCount} lines)`);
}

// ─── Check header guards ───
console.log('\n📁 Header Guard Check');
const headerGuardPairs = {
  'include/voxcpm.h': ['VOXCPM_H', 'VOXCPM_H'],
  'include/tensor.h': ['TENSOR_H', 'TENSOR_H'],
  'include/nn.h': ['NN_H', 'NN_H'],
  'include/model.h': ['MODEL_H', 'MODEL_H'],
  'include/platform.h': ['PLATFORM_H', 'PLATFORM_H'],
  'include/audio.h': ['AUDIO_H', 'AUDIO_H'],
  'include/sampler.h': ['SAMPLER_H', 'SAMPLER_H'],
};
for (const [path, [guard]] of Object.entries(headerGuardPairs)) {
  const fullPath = join(ROOT, path);
  if (!existsSync(fullPath)) continue;
  const content = readFileSync(fullPath, 'utf-8');
  const hasIfndef = content.includes(`#ifndef ${guard}`);
  const hasDefine = content.includes(`#define ${guard}`);
  const hasEndif = content.includes(`#endif /* ${guard} */`) || content.includes(`#endif // ${guard}`) || content.includes(`#endif`);
  if (hasIfndef && hasDefine && hasEndif) {
    check(true, `${path} — guard: ${guard}`);
  } else {
    check(false, `${path} — guard: ${guard}`, `(ifndef:${hasIfndef} define:${hasDefine} endif:${hasEndif})`);
  }
}

// ─── Phase 1: Weight format & loading ───
console.log('\n📁 Phase 1: Model Loading');
checkFile('docs/WEIGHT_FORMAT.md', 'Weight format spec');
checkFile('scripts/convert_weights.py', 'Weight converter script');
checkIncludesHeader('include/model.h', 'VxcpmHeader');
checkIncludesHeader('include/model.h', 'VxcpmTensorMeta');
checkIncludesHeader('include/model.h', 'WeightIndex');
checkIncludesHeader('include/model.h', 'WeightEntry');
checkIncludesHeader('include/model.h', 'weight_index_build');
checkIncludesHeader('include/model.h', 'weight_index_find');
checkIncludesHeader('include/model.h', 'weight_load_tensor');
checkIncludesHeader('src/model.c', 'weight_index_build');
checkIncludesHeader('src/model.c', 'weight_index_find');
checkIncludesHeader('src/model.c', 'weight_load_tensor');
checkIncludesHeader('src/model.c', 'crc32c');
checkFileSize('src/model.c', 700);

// ─── Check CMakeLists.txt source file listing ───
console.log('\n📁 CMake Source File Coverage');
const cmakeContent = readFileSync(join(ROOT, 'CMakeLists.txt'), 'utf-8');
for (const f of sourceFiles) {
  const fileName = f.replace('src/', '');
  check(cmakeContent.includes(fileName), `CMakeLists.txt lists ${fileName}`);
}

// ─── Summary ───
console.log('\n══════════════════════════════════════════════');
console.log(` Results: ${passed} passed, ${failed} failed, ${warnings} warnings`);
console.log('══════════════════════════════════════════════\n');

if (failed > 0) {
  console.log('⚠️  Some checks failed. Review details above.');
  process.exit(1);
} else {
  console.log('✅ All checks passed. Phase 0 structure is complete.');
  process.exit(0);
}
