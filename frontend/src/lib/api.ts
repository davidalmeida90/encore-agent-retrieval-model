import { request, type RequestOptions } from '@/lib/http'
import { supabase } from '@/lib/supabase'

export async function getAccessToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token ?? null
}

async function withAuth<T>(
  path: string,
  options: Omit<RequestOptions, 'accessToken'> = {},
): Promise<T> {
  return request<T>(path, {
    ...options,
    accessToken: await getAccessToken(),
  })
}

export const api = {
  get: <T>(path: string, options?: Omit<RequestOptions, 'accessToken' | 'method' | 'body'>) =>
    withAuth<T>(path, { ...options, method: 'GET' }),

  post: <T>(
    path: string,
    body?: unknown,
    options?: Omit<RequestOptions, 'accessToken' | 'method' | 'body'>,
  ) => withAuth<T>(path, { ...options, method: 'POST', body }),

  put: <T>(
    path: string,
    body?: unknown,
    options?: Omit<RequestOptions, 'accessToken' | 'method' | 'body'>,
  ) => withAuth<T>(path, { ...options, method: 'PUT', body }),

  patch: <T>(
    path: string,
    body?: unknown,
    options?: Omit<RequestOptions, 'accessToken' | 'method' | 'body'>,
  ) => withAuth<T>(path, { ...options, method: 'PATCH', body }),

  delete: <T>(path: string, options?: Omit<RequestOptions, 'accessToken' | 'method' | 'body'>) =>
    withAuth<T>(path, { ...options, method: 'DELETE' }),
}
