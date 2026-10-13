/** HTTP client works in browser and Next.js. Types are the integration aid;
 * OpenAPI and Python models are authoritative. Regenerate types from OpenAPI later. */
export type Method = 'clip' | 'bm25';
export type Split = 'train' | 'val' | 'test' | 'unspecified';
export interface Caption { text: string; language: 'en' | 'vi' | 'und' }
export interface ImagePublic {
  id: string; dataset: string; split: Split; width: number; height: number;
  captions: Caption[]; image_url: string; thumbnail_url: string;
}
export interface SearchRequest {
  query: string; method?: Method; top_k?: number; offset?: number; limit?: number;
  min_similarity?: number | null; dataset?: string | null; split?: Split | null;
}
export interface SearchResponse {
  request_id: string; api_version: '2.0.0'; mode: 'mock' | 'real'; method: Method;
  score_type: 'cosine_similarity' | 'bm25_score'; query: string | null;
  top_k: number; offset: number; limit: number; total: number; has_more: boolean;
  next_offset: number | null; index_version: string;
  results: { rank: number; image: ImagePublic; score: number; similarity: number | null }[];
}
export interface SuccessResponse<T> {
  success: true; statusCode: 200; data: T; timestamp: string;
}
export interface ErrorResponse {
  success: false; statusCode: number; message: string[]; path: string; timestamp: string;
}
export interface Health {
  status: 'ok' | 'not_ready'; mode: 'mock' | 'real'; ready: boolean;
  model_loaded: boolean; index_loaded: boolean; api_version: '2.0.0';
}
export interface SimilarRequest {
  top_k?: number; offset?: number; limit?: number; min_similarity?: number | null;
}
export class ApiError extends Error {
  constructor(public status: number, public payload: ErrorResponse,
    public requestId: string | null = null) {
    super(payload.message.join('; '));
  }
}
async function requestJson<T>(baseUrl: string, path: string, init: RequestInit = {}):
  Promise<SuccessResponse<T>> {
  const response = await fetch(new URL(path, baseUrl), {
    ...init,
    headers: { Accept: 'application/json',
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
    },
  });
  const payload = await response.json() as SuccessResponse<T> | ErrorResponse;
  if (!response.ok || !payload.success) {
    throw new ApiError(response.status, payload as ErrorResponse,
      response.headers.get('X-Request-ID'));
  }
  // Compile-time cast only; runtime guarantee comes from the validated API.
  return payload;
}
export function searchImages(baseUrl: string, body: SearchRequest,
  signal?: AbortSignal): Promise<SuccessResponse<SearchResponse>> {
  return requestJson(baseUrl, '/search', { method: 'POST', body: JSON.stringify(body), signal });
}
export function getHealth(baseUrl: string, signal?: AbortSignal): Promise<SuccessResponse<Health>> {
  return requestJson(baseUrl, '/health', { signal });
}
export function getImage(baseUrl: string, imageId: string,
  signal?: AbortSignal): Promise<SuccessResponse<ImagePublic>> {
  return requestJson(baseUrl, `/images/${encodeURIComponent(imageId)}`, { signal });
}
export function getSimilarImages(baseUrl: string, imageId: string, filters: SimilarRequest = {},
  signal?: AbortSignal): Promise<SuccessResponse<SearchResponse>> {
  const query = new URLSearchParams();
  for (const key of ['top_k', 'offset', 'limit', 'min_similarity'] as const) {
    const value = filters[key];
    if (value !== null && value !== undefined) query.set(key, String(value));
  }
  const suffix = query.size ? `?${query.toString()}` : '';
  return requestJson(baseUrl, `/images/${encodeURIComponent(imageId)}/similar${suffix}`, { signal });
}
export const imageUrl = (baseUrl: string, path: string): string => new URL(path, baseUrl).href;
