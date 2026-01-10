import { type IGatsbyImageData } from 'gatsby-plugin-image'

export const getRefinedImage = (heroImage: IGatsbyImageData | undefined) => {
  if (heroImage === undefined) throw new Error('이미지가 존재하지 않습니다.')

  return heroImage
}

export const getRefinedStringValue = (value: string | null | undefined): string => {
  if (value === null || value === undefined) {
    if (typeof window === 'undefined') {
      return ''
    }
    throw new Error('값이 존재하지 않습니다.')
  }

  return value
}
