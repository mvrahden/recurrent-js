# Class: `Utils`

* Random Number Functions
  * `static randf(min: number, max: number): number`
  * `static randi(min: number, max: number): number`
  * `static randn(mu: number, std: number): number`
  * `static skewedRandn(mu: number, std: number, skew: number): number`
* Statistic Tools
  * `static sum(arr: Array<number> | Float64Array): number`
  * `static mean(arr: Array<number> | Float64Array): number`
  * `static median(arr: Array<number>): number`
  * `static mode(arr: Array<number>, precision?: number): Array<number>`
  * `static var(arr: Array<number>, normalization?: 'uncorrected' | 'biased' | 'unbiased'): number`
  * `static std(arr: Array<number>, normalization?: 'uncorrected' | 'biased' | 'unbiased'): number`
* Array Filler
  * `static fillRandn(arr: Array<number> | Float64Array, mu: number, std: number): void`
  * `static fillRand(arr: Array<number> | Float64Array, min: number, max: number): void`
  * `static fillConst(arr: Array<number> | Float64Array, c: number): void`
  * `static fillArray(n: number, val: number): Array<number> | Float64Array`
* Array Creation
  * `static zeros(n: number): Array<number> | Float64Array`
  * `static ones(n: number): Array<number> | Float64Array`
  * `static argmax(arr: Array<number> | Float64Array): number`
  * `static sampleWeighted(arr: Array<number> | Float64Array): number`
