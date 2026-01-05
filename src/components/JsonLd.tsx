const SCHEMA_CONTEXT = 'https://schema.org'
const DEFAULT_LANGUAGE = 'ko-KR'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type JsonLdData = Record<string, any>

export interface JsonLdProps {
  data: JsonLdData | JsonLdData[]
}

export interface SchemaBase {
  '@context': typeof SCHEMA_CONTEXT
  inLanguage: typeof DEFAULT_LANGUAGE
}

export interface PersonReference {
  '@type': 'Person'
  name: string
  url?: string
}

export interface OrganizationReference {
  '@type': 'Organization'
  name: string
  url?: string
}

export interface WebSiteSchemaInput {
  siteUrl: string
  siteName: string
  description: string
}

export interface SearchAction {
  '@type': 'SearchAction'
  target: string
  'query-input': string
}

export interface WebSiteSchema extends SchemaBase {
  '@type': 'WebSite'
  url: string
  name: string
  description: string
  potentialAction: SearchAction
}

export interface BlogPostingSchemaInput {
  title: string
  description: string
  datePublished: string
  dateModified: string
  url: string
  image: string
  authorName: string
  authorUrl?: string
  tags?: string[]
}

export interface BlogPostingSchema extends SchemaBase {
  '@type': 'BlogPosting'
  headline: string
  description: string
  datePublished: string
  dateModified: string
  url: string
  image?: string
  author: PersonReference
  keywords?: string[]
}

export interface ArticleSchemaInput {
  title: string
  description: string
  datePublished: string
  dateModified: string
  url: string
  image: string
  authorName: string
  authorUrl?: string
  tags?: string[]
}

export interface ArticleSchema extends SchemaBase {
  '@type': 'Article'
  headline: string
  description: string
  datePublished: string
  dateModified: string
  url: string
  image?: string
  author: PersonReference
  keywords?: string[]
}

export type ReviewMediaType = 'movie' | 'series' | 'animation' | 'book'

export interface ItemReviewedInput {
  name: string
  url?: string
  image?: string
  creatorName?: string
  creatorUrl?: string
}

export interface ItemReviewedSchema {
  '@type': 'Movie' | 'TVSeries' | 'CreativeWork' | 'Book'
  name: string
  url?: string
  image?: string
  genre?: string
  author?: PersonReference
  creator?: PersonReference
}

export interface RatingSchema {
  '@type': 'Rating'
  ratingValue: number
  bestRating: number
  worstRating: number
}

export interface ReviewSchemaInput {
  title: string
  description: string
  datePublished: string
  url: string
  image: string
  rating: number
  mediaType: ReviewMediaType
  authorName: string
  authorUrl?: string
  itemReviewed: ItemReviewedInput
}

export interface ReviewSchema extends SchemaBase {
  '@type': 'Review'
  name: string
  description: string
  datePublished: string
  url: string
  image?: string
  reviewRating: RatingSchema
  author: PersonReference
  itemReviewed: ItemReviewedSchema
}

export interface PersonSchemaInput {
  name: string
  alternateName?: string
  url?: string
  image?: string
  jobTitle?: string
  worksFor?: string | { name: string; url?: string }
  sameAs?: string[]
}

export interface PersonSchema extends SchemaBase {
  '@type': 'Person'
  name: string
  alternateName?: string
  url?: string
  image?: string
  jobTitle?: string
  worksFor?: OrganizationReference
  sameAs?: string[]
}

export interface BreadcrumbItemInput {
  name: string
  url: string
}

export interface BreadcrumbListItem {
  '@type': 'ListItem'
  position: number
  name: string
  item: {
    '@id': string
    name: string
  }
}

export interface BreadcrumbSchema extends SchemaBase {
  '@type': 'BreadcrumbList'
  itemListElement: BreadcrumbListItem[]
}

const createPersonReference = (name: string, url?: string): PersonReference => ({
  '@type': 'Person',
  name,
  ...(url ? { url } : {})
})

const createOrganizationReference = (name: string, url?: string): OrganizationReference => ({
  '@type': 'Organization',
  name,
  ...(url ? { url } : {})
})

const createItemReviewedSchema = (mediaType: ReviewMediaType, itemReviewed: ItemReviewedInput): ItemReviewedSchema => {
  const typeMap: Record<ReviewMediaType, ItemReviewedSchema['@type']> = {
    movie: 'Movie',
    series: 'TVSeries',
    animation: 'CreativeWork',
    book: 'Book'
  }

  const creator = itemReviewed.creatorName
    ? createPersonReference(itemReviewed.creatorName, itemReviewed.creatorUrl)
    : undefined

  const base: ItemReviewedSchema = {
    '@type': typeMap[mediaType],
    name: itemReviewed.name,
    ...(itemReviewed.url ? { url: itemReviewed.url } : {}),
    ...(itemReviewed.image ? { image: itemReviewed.image } : {}),
    ...(mediaType === 'animation' ? { genre: 'Animation' } : {})
  }

  if (creator) {
    if (mediaType === 'book') {
      base.author = creator
    } else {
      base.creator = creator
    }
  }

  return base
}

export const JsonLd = ({ data }: JsonLdProps) => (
  <script
    type="application/ld+json"
    dangerouslySetInnerHTML={{ __html: JSON.stringify(data) }}
  />
)

export const createWebSiteSchema = ({ siteUrl, siteName, description }: WebSiteSchemaInput): WebSiteSchema => {
  const normalizedUrl = siteUrl.replace(/\/$/, '')

  return {
    '@context': SCHEMA_CONTEXT,
    '@type': 'WebSite',
    inLanguage: DEFAULT_LANGUAGE,
    url: normalizedUrl,
    name: siteName,
    description,
    potentialAction: {
      '@type': 'SearchAction',
      target: `${normalizedUrl}/?q={search_term_string}`,
      'query-input': 'required name=search_term_string'
    }
  }
}

export const createBlogPostingSchema = ({
  title,
  description,
  datePublished,
  dateModified,
  url,
  image,
  authorName,
  authorUrl,
  tags
}: BlogPostingSchemaInput): BlogPostingSchema => ({
  '@context': SCHEMA_CONTEXT,
  '@type': 'BlogPosting',
  inLanguage: DEFAULT_LANGUAGE,
  headline: title,
  description,
  datePublished,
  dateModified,
  url,
  ...(image ? { image } : {}),
  author: createPersonReference(authorName, authorUrl),
  ...(tags && tags.length > 0 ? { keywords: tags } : {})
})

export const createArticleSchema = ({
  title,
  description,
  datePublished,
  dateModified,
  url,
  image,
  authorName,
  authorUrl,
  tags
}: ArticleSchemaInput): ArticleSchema => ({
  '@context': SCHEMA_CONTEXT,
  '@type': 'Article',
  inLanguage: DEFAULT_LANGUAGE,
  headline: title,
  description,
  datePublished,
  dateModified,
  url,
  ...(image ? { image } : {}),
  author: createPersonReference(authorName, authorUrl),
  ...(tags && tags.length > 0 ? { keywords: tags } : {})
})

export const createReviewSchema = ({
  title,
  description,
  datePublished,
  url,
  image,
  rating,
  mediaType,
  authorName,
  authorUrl,
  itemReviewed
}: ReviewSchemaInput): ReviewSchema => ({
  '@context': SCHEMA_CONTEXT,
  '@type': 'Review',
  inLanguage: DEFAULT_LANGUAGE,
  name: title,
  description,
  datePublished,
  url,
  ...(image ? { image } : {}),
  reviewRating: {
    '@type': 'Rating',
    ratingValue: rating,
    bestRating: 5,
    worstRating: 1
  },
  author: createPersonReference(authorName, authorUrl),
  itemReviewed: createItemReviewedSchema(mediaType, itemReviewed)
})

export const createPersonSchema = ({
  name,
  alternateName,
  url,
  image,
  jobTitle,
  worksFor,
  sameAs
}: PersonSchemaInput): PersonSchema => {
  const organization = typeof worksFor === 'string'
    ? createOrganizationReference(worksFor)
    : worksFor
      ? createOrganizationReference(worksFor.name, worksFor.url)
      : undefined

  return {
    '@context': SCHEMA_CONTEXT,
    '@type': 'Person',
    inLanguage: DEFAULT_LANGUAGE,
    name,
    ...(alternateName ? { alternateName } : {}),
    ...(url ? { url } : {}),
    ...(image ? { image } : {}),
    ...(jobTitle ? { jobTitle } : {}),
    ...(organization ? { worksFor: organization } : {}),
    ...(sameAs && sameAs.length > 0 ? { sameAs } : {})
  }
}

export const createBreadcrumbSchema = (items: BreadcrumbItemInput[]): BreadcrumbSchema => ({
  '@context': SCHEMA_CONTEXT,
  '@type': 'BreadcrumbList',
  inLanguage: DEFAULT_LANGUAGE,
  itemListElement: items.map((item, index) => ({
    '@type': 'ListItem',
    position: index + 1,
    name: item.name,
    item: {
      '@id': item.url,
      name: item.name
    }
  }))
})
