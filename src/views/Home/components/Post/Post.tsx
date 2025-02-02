import { Link, graphql, useStaticQuery } from "gatsby";
import {
  GatsbyImage,
  type IGatsbyImageData,
  getImage,
} from "gatsby-plugin-image";
import { match } from "ts-pattern";

import { getRefinedImage } from "@/utils";

import { Date, Description, TagList, Title } from "./components";
import * as styles from "./Post.module.scss";

type PostProps = {
  variants: "card" | "item";
  title: string;
  description: string;
  date: string;
  tags: readonly string[];
  slug: string;
  heroImage: IGatsbyImageData | undefined;
  heroImageAlt: string | undefined | null;
};

export const Post = ({
  variants,
  title,
  description,
  date,
  tags,
  slug,
  heroImage,
  heroImageAlt,
}: PostProps) => {
  const defaultImage = useStaticQuery(graphql`
    query {
      cover: file(relativePath: { eq: "cover.png" }) {
        childImageSharp {
          gatsbyImageData(placeholder: BLURRED)
        }
      }
    }
  `);

  const image = getRefinedImage(
    heroImage === undefined ? getImage(defaultImage.cover) : heroImage,
  );
  const imageAlt = heroImageAlt ?? "Cover Image";

  return (
    <Link to={`/posts${slug}`} className={styles.articleLink}>
      {match(variants)
        .with("card", () => (
          <article className={styles.card}>
            <figure>
              <GatsbyImage
                image={image}
                alt={imageAlt}
                className={styles.cardImage}
              />
              <figcaption className={styles.cardCaption}>
                <Date date={date} className={styles.cardDate} />
                <TagList tags={tags} className={styles.cardTagList} />
                <Title title={title} className={styles.cardTitle} />
                <Description
                  description={description}
                  className={styles.cardDescription}
                />
              </figcaption>
            </figure>
          </article>
        ))
        .with("item", () => (
          <article className={styles.item}>
            <figure className={styles.itemFigure}>
              <GatsbyImage
                image={image}
                alt={imageAlt}
                className={styles.itemImage}
              />
              <figcaption className={styles.itemCaption}>
                <Title title={title} className={styles.itemTitle} />
                <Description
                  description={description}
                  className={styles.itemDescription}
                />
                <Date date={date} className={styles.itemDate} />
                <TagList tags={tags} className={styles.itemTags} />
              </figcaption>
            </figure>
          </article>
        ))
        .exhaustive()}
    </Link>
  );
};
