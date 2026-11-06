import { test } from 'node:test';
import assert from 'node:assert';

import { clampRegion } from './region.js';

test('clampRegion with valid range', () => {
  const result = clampRegion(1.0, 3.0, 5.0);
  assert.strictEqual(result, null); // No error
});

test('clampRegion with inverted range', () => {
  const result = clampRegion(3.0, 1.0, 5.0);
  assert.ok(result.includes('Inverted range'));
});

test('clampRegion with start out of bounds', () => {
  const result = clampRegion(-1.0, 2.0, 5.0);
  assert.ok(result.includes('extends past audio boundaries'));
});

test('clampRegion with end out of bounds', () => {
  const result = clampRegion(2.0, 6.0, 5.0);
  assert.ok(result.includes('extends past audio boundaries'));
});
