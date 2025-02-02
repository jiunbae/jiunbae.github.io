import clsx from "clsx";

import { TagButton } from "../TagButton";
import * as styles from "./TagList.module.scss";

type TagListProps = {
  tags: readonly string[];
  className?: string;
};

export const TagList = ({ tags, className }: TagListProps) => (
  <ul className={clsx(styles.tagList, className)}>
    {tags.map((tag) => (
      <TagButton key={tag} name={tag}></TagButton>
    ))}
  </ul>
);
