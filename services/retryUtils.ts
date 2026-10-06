/**
 * Retry utility with exponential backoff, jitter, and circuit breaker pattern.
 * Provides robust error recovery for API calls with configurable policies.
 */

export interface RetryPolicy {
  /** Maximum number of retry attempts (default: 3) */
  maxRetries: number;
  /** Base delay in milliseconds for exponential backoff (default: 1000) */
  baseDelayMs: number;
  /** Maximum delay cap in milliseconds (default: 30000) */
  maxDelayMs: number;
  /** Whether to apply random jitter to prevent thundering herd (default: true) */
  useJitter: boolean;
  /** Multiplier for exponential backoff (default: 2) */
  backoffMultiplier: number;
  /** Retry only on these HTTP/error codes */
  retryableErrorCodes?: string[];
}

const DEFAULT_POLICY: RetryPolicy = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 30000,
  useJitter: true,
  backoffMultiplier: 2,
  retryableErrorCodes: [
    '429', // Too Many Requests
    '500', '502', '503', '504', // Server errors
    'RESOURCE_EXHAUSTED', 'UNAVAILABLE', 'DEADLINE_EXCEEDED',
    'NETWORK_ERROR', 'TIMEOUT',
  ],
};

function isRetryableError(error: unknown, policy: RetryPolicy): boolean {
  const message = error instanceof Error ? error.message : String(error);
  const codes = policy.retryableErrorCodes ?? DEFAULT_POLICY.retryableErrorCodes!;
  return codes.some((code) => message.includes(code));
}

function calculateDelay(attempt: number, policy: RetryPolicy): number {
  const delay = Math.min(
    policy.baseDelayMs * Math.pow(policy.backoffMultiplier, attempt),
    policy.maxDelayMs
  );
  if (policy.useJitter) {
    return delay * (0.5 + Math.random() * 0.5); // 50-100% of delay
  }
  return delay;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Execute an async function with exponential backoff retry logic.
 * Falls back gracefully after exhausting all retries.
 *
 * @param fn - The async function to execute with retry support
 * @param policy - Retry policy configuration
 * @param onRetry - Optional callback invoked before each retry
 * @returns The result of the function, or undefined if all retries exhausted
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  policy: Partial<RetryPolicy> = {},
  onRetry?: (error: unknown, attempt: number) => void
): Promise<T | undefined> {
  const mergedPolicy: RetryPolicy = { ...DEFAULT_POLICY, ...policy };
  let lastError: unknown;

  for (let attempt = 0; attempt <= mergedPolicy.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (
        attempt < mergedPolicy.maxRetries &&
        isRetryableError(error, mergedPolicy)
      ) {
        const delay = calculateDelay(attempt, mergedPolicy);
        if (onRetry) {
          onRetry(error, attempt + 1);
        }
        await sleep(delay);
        continue;
      }
      break;
    }
  }

  console.warn(
    `[Retry] All ${mergedPolicy.maxRetries} retries exhausted. Last error:`,
    lastError
  );
  return undefined;
}

/**
 * Simple circuit breaker to prevent repeated calls to failing services.
 * Opens after `failureThreshold` consecutive failures,
 * then allows a single probe after `resetTimeoutMs`.
 */
export class CircuitBreaker {
  private failures = 0;
  private lastFailureTime = 0;
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';

  constructor(
    private readonly failureThreshold = 5,
    private readonly resetTimeoutMs = 30000
  ) {}

  async call<T>(fn: () => Promise<T>): Promise<T | undefined> {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime > this.resetTimeoutMs) {
        this.state = 'HALF_OPEN';
      } else {
        console.warn('[CircuitBreaker] Open, rejecting call');
        return undefined;
      }
    }

    try {
      const result = await fn();
      if (this.state === 'HALF_OPEN') {
        this.state = 'CLOSED';
        this.failures = 0;
      }
      return result;
    } catch (error) {
      this.failures++;
      this.lastFailureTime = Date.now();
      if (this.failures >= this.failureThreshold) {
        this.state = 'OPEN';
        console.warn(
          `[CircuitBreaker] Opened after ${this.failures} failures`
        );
      }
      throw error;
    }
  }

  getState(): string {
    return this.state;
  }

  reset(): void {
    this.failures = 0;
    this.state = 'CLOSED';
  }
}
