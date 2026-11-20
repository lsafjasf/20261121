import assert from "node:assert/strict";
import test from "node:test";
import { runTimingChecks } from "../src/verification.js";

test("all in-app timing checks pass for both generated examples", () => {
  const examples = [
    { sourceBpm: 88, sourceIn: 2.727272727272727, sourceOut: 8.181818181818182 },
    { sourceBpm: 124, sourceIn: 0, sourceOut: 3.870967741935484 },
  ];
  for (const example of examples) {
    const results = runTimingChecks({
      ...example,
      ladder: [68, 72, 76, 80, 84, 88, 92, 96, 100, 104],
    });
    assert.equal(results.length, 3);
    assert.ok(results.every((result) => result.pass), JSON.stringify(results, null, 2));
  }
});
