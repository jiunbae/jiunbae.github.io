import * as styles from "./About.module.scss";
import { ProfileCard } from "@/components";

const AboutPage = () => {
  return (
    <main className={styles.page}>
      <aside className={styles.aside}>
        <ProfileCard />
      </aside>
      <h1 className={styles.heading}>About page</h1>
      <p className={styles.paragraph}>About page</p>
    </main>
  );
};

export default AboutPage;

export const Head = () => <title>About</title>;
