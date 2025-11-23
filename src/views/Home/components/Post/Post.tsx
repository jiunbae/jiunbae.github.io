import { Link, graphql, useStaticQuery } from 'gatsby'
import { GatsbyImage, type IGatsbyImageData, getImage } from 'gatsby-plugin-image'

import { getRefinedImage } from '@/utils'

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
  heroImage: IGatsbyImageData;
  heroImageAlt: string;
  className: string;
}

type CardPostProps = Omit<PostProps, 'variants'> & StyledProps
type ItemPostProps = Omit<PostProps, 'variants'> & StyledProps

export const CardPost = ({
  title,
  description,
  date,
  tags,
  slug,
  heroImage,
  heroImageAlt,
  className
}: CardPostProps) => (
  <Link to={`/posts${slug}`} className={className}>
    <article className={styles.card}>
      <figure>
        <GatsbyImage
          image={heroImage}
          alt={heroImageAlt}
          className={styles.cardImage}
          loading="lazy"
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
  className
}: ItemPostProps) => (
  <Link to={`/posts${slug}`} className={className}>
    <article className={styles.item}>
      <figure className={styles.itemFigure}>
        <GatsbyImage
          image={heroImage}
          alt={heroImageAlt}
          className={styles.itemImage}
          loading="lazy"
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
  const defaultImage = useStaticQuery(graphql`
    query {
      cover: file(relativePath: { eq: "cover.png" }) {
        childImageSharp {
          gatsbyImageData(placeholder: BLURRED)
        }
      }
    }
  `)

  const image = getRefinedImage(heroImage === undefined ? getImage(defaultImage.cover) : heroImage)
  const imageAlt = heroImageAlt ?? 'Cover Image'

  if (variants === 'card') {
    return (
      <CardPost
        title={title}
        description={description}
        date={date}
        tags={tags}
        slug={slug}
        heroImage={image}
        heroImageAlt={imageAlt}
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
        heroImage={image}
        heroImageAlt={imageAlt}
        className={styles.articleLink}
      />
    )
  }

  return null
}
