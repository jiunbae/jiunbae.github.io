import { publicServiceCatalog } from '@/data/serviceCatalog'

export function GET() {
  return new Response(JSON.stringify(publicServiceCatalog()), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  })
}
