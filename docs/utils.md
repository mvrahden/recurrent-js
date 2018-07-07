# Class: `Utils`

The class `Utils` provides a collection of useful statistical tools, evaluation tools and array tools.

## Class Structure

* Random Number Functions
  * `[static] randf(min: number, max: number): number`
  * `[static] randi(min: number, max: number): number`
  * `[static] randn(mu: number, std: number): number`
  * `[static] skewedRandn(mu: number, std: number, skew: number): number`
* Statistic Tools
  * `[static] sum(arr: Array<number> | Float64Array): number`
  * `[static] mean(arr: Array<number> | Float64Array): number`
  * `[static] median(arr: Array<number> | Float64Array): number`
  * `[static] mode(arr: Array<number> | Float64Array, precision?: number): Array<number>`
  * `[static] var(arr: Array<number> | Float64Array, normalization?: 'uncorrected' | 'biased' | 'unbiased'): number`
  * `[static] std(arr: Array<number> | Float64Array, normalization?: 'uncorrected' | 'biased' | 'unbiased'): number`
* Array Filler
  * `[static] fillRandn(arr: Array<number> | Float64Array, mu: number, std: number): void`
  * `[static] fillRand(arr: Array<number> | Float64Array, min: number, max: number): void`
  * `[static] fillConst(arr: Array<number> | Float64Array, c: number): void`
  * `[static] fillArray(n: number, val: number): Array<number> | Float64Array`
* Array Creation
  * `[static] zeros(n: number): Array<number> | Float64Array`
  * `[static] ones(n: number): Array<number> | Float64Array`
* Output Functions
  * `[static] softmax(arr: Array<number> | Float64Array): Array<number> | Float64Array`
  * `[static] argmax(arr: Array<number> | Float64Array): number`
  * `[static] sampleWeighted(arr: Array<number> | Float64Array): number`

## Usage

Import the `Utils`-class and use the provided **static** methods.
Each method comes with its respective description.

```typescript
import { Utils } from 'recurrent-js';

const randomFloat = Utils.randf(0, 10);
const randomInt = Utils.randi(0, 10);
const randomNormal = Utils.randi(0, 1);

const sum = Utils.sum([1, 2, 4, 10]);
```