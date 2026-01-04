import { Link } from 'gatsby'
import { GatsbyImage, type IGatsbyImageData } from 'gatsby-plugin-image'

import { sanitizePostSlug } from '@/utils'

import { Date, Description, TagList, Title } from './components'
import * as styles from './Post.module.scss'

type PostProps = {
  variants: 'card' | 'item';
  title: string;
  description: string;
  date: string;
  tags: readonly string[];
  slug: string;
  heroImage: IGatsbyImageData | undefined;
  heroImageAlt: string | undefined | null;
};

type StyledProps = {
  heroImage: IGatsbyImageData | undefined;
  heroImageAlt: string;
  ogImageSrc: string;
  className: string;
}

type CardPostProps = Omit<PostProps, 'variants'> & StyledProps
type ItemPostProps = Omit<PostProps, 'variants'> & StyledProps

const PostImage = ({
  heroImage,
  heroImageAlt,
  ogImageSrc,
  className
}: {
  heroImage: IGatsbyImageData | undefined;
  heroImageAlt: string;
  ogImageSrc: string;
  className: string;
}) => {
  if (heroImage) {
    return (
      <GatsbyImage
        image={heroImage}
        alt={heroImageAlt}
        className={className}
        loading="lazy"
      />
    )
  }

  return (
    <div className={className}>
      <img
        src={ogImageSrc}
        alt={heroImageAlt}
        loading="lazy"
        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
      />
    </div>
  )
}

export const CardPost = ({
  title,
  description,
  date,
  tags,
  slug,
  heroImage,
  heroImageAlt,
  ogImageSrc,
  className
}: CardPostProps) => (
  <Link to={`/posts${slug}`} className={className}>
    <article className={styles.card}>
      <figure>
        <PostImage
          heroImage={heroImage}
          heroImageAlt={heroImageAlt}
          ogImageSrc={ogImageSrc}
          className={styles.cardImage}
        />
        <figcaption className={styles.cardCaption}>
          <Date date={date} className={styles.cardDate} />
          <TagList tags={tags} className={styles.cardTagList} />
          <Title title={title} className={styles.cardTitle} />
          <Description description={description} className={styles.cardDescription} />
        </figcaption>
      </figure>
    </article>
  </Link>
)

export const ItemPost = ({
  title,
  description,
  date,
  tags,
  slug,
  heroImage,
  heroImageAlt,
  ogImageSrc,
  className
}: ItemPostProps) => (
  <Link to={`/posts${slug}`} className={className}>
    <article className={styles.item}>
      <figure className={styles.itemFigure}>
        <PostImage
          heroImage={heroImage}
          heroImageAlt={heroImageAlt}
          ogImageSrc={ogImageSrc}
          className={styles.itemImage}
        />
        <figcaption className={styles.itemCaption}>
          <Title title={title} className={styles.itemTitle} />
          <Description description={description} className={styles.itemDescription} />
          <Date date={date} className={styles.itemDate} />
          <TagList tags={tags} className={styles.itemTags} />
        </figcaption>
      </figure>
    </article>
  </Link>
)

export const Post = ({ variants, title, description, date, tags, slug, heroImage, heroImageAlt }: PostProps) => {
  const ogImageSrc = `/og/posts/${sanitizePostSlug(slug)}.png`
  const imageAlt = heroImageAlt ?? title

  if (variants === 'card') {
    return (
      <CardPost
        title={title}
        description={description}
        date={date}
        tags={tags}
        slug={slug}
        heroImage={heroImage}
        heroImageAlt={imageAlt}
        ogImageSrc={ogImageSrc}
        className={styles.articleLink}
      />
    )
  }

  if (variants === 'item') {
    return (
      <ItemPost
        title={title}
        description={description}
        date={date}
        tags={tags}
        slug={slug}
        heroImage={heroImage}
        heroImageAlt={imageAlt}
        ogImageSrc={ogImageSrc}
        className={styles.articleLink}
      />
    )
  }

  return null
}
