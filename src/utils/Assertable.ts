/**
 * Adds static assertability to Classes
 */
export abstract class Assertable {
  /**
   * Asserts a condition and throws Error of not <i>truthy</i>
   * @param condition 
   * @param message 
   */
  protected static assert(condition, message = '') {
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
