import { Description, Heading, IconList } from "./components";
import * as styles from "./ProfileCard.module.scss";

export const ProfileCard = () => {
  return (
    <div className={styles.card}>
      <Heading text="Jiunbae" />
      <Description />
      <div className={styles.info}>
        <IconList />
      </div>
    </div>
  );
};
