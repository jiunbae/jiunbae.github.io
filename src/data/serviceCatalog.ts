import services from './services'

/** Published once by the blog, bundled by jiun-api; never contains registry credentials. */
export function publicServiceCatalog() {
  const absolute = (value: string | undefined) => value ? new URL(value, 'https://jiun.dev').href : null
  return {
    schemaVersion: 1,
    services: [
      { id: 'blog', name: 'jiun.dev · Blog', description: '글과 노트, 프로젝트와 서비스 소식을 모은 블로그.', url: 'https://jiun.dev/', status: 'live', kind: 'page', iconUrl: 'https://jiun.dev/logo-icon.svg', iconText: null },
      ...services.filter(service => service.status !== 'internal').map(service => ({
        id: service.id, name: service.title, description: service.description,
        url: absolute(service.slug ?? service.github ?? (service.npm ? `https://www.npmjs.com/package/${service.npm}` : service.post)),
        status: service.status, kind: service.kind ?? 'service',
        iconUrl: service.iconType === 'image' ? absolute(service.icon) : null,
        iconText: service.iconType === 'letter' ? service.icon : null,
      })),
    ],
  }
}
