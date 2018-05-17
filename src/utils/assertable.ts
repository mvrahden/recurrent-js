/**
 * Adds static assertability to Classes
 */
export abstract class Assertable {
  /**
   * Asserts a condition and throws Error if not assertion fails
   * @param {boolean} condition 
   * @param {string} message 
   */
  protected static assert(condition: boolean, message: string = '') {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }
}
