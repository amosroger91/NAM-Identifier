export function clampRegion(startSec, endSec, durationSec) {
  // Check for inverted range
  if (startSec >= endSec) {
    return "Inverted range: start must be less than end";
  }
  
  // Check if range is outside the duration
  if (startSec < 0 || endSec > durationSec) {
    return `Range extends past audio boundaries: ${startSec} to ${endSec} with duration ${durationSec}`;
  }
  
  // Valid range
  return null; // No error
}