import type { HeadProps } from 'gatsby'
import { Seo } from '@/components'

import * as styles from './Contact.module.scss'

const ContactPage = () => {
  return (
    <main className={styles.page}>
      <div className={styles.content}>
        <h1>문의</h1>
        
        <section className={styles.introduction}>
          <p>
            안녕하세요! 기술 블로그 jiun.dev에 방문해주셔서 감사합니다.
            AI/ML 개발, 블로그 콘텐츠, 기술 관련 문의사항이 있으시면 아래 연락처로 편하게 연락 주세요.
          </p>
        </section>

        <section className={styles.contactInfo}>
          <h2>연락처</h2>
          <div className={styles.contactGrid}>
            <div className={styles.contactItem}>
              <h3>📧 이메일</h3>
              <p><a href="mailto:contact@jiun.dev">contact@jiun.dev</a></p>
              <p className={styles.description}>
                기술 문의, 블로그 피드백, 협업 문의 등 모든 내용 환영합니다.
              </p>
            </div>
            
            <div className={styles.contactItem}>
              <h3>💼 LinkedIn</h3>
              <p><a href="https://linkedin.com/in/jiunbae" target="_blank" rel="noopener noreferrer">linkedin.com/in/jiunbae</a></p>
              <p className={styles.description}>
                프로페셔널 네트워킹 및 경력 관련 문의
              </p>
            </div>
            
            <div className={styles.contactItem}>
              <h3>🐙 GitHub</h3>
              <p><a href="https://github.com/jiunbae" target="_blank" rel="noopener noreferrer">github.com/jiunbae</a></p>
              <p className={styles.description}>
                오픈소스 프로젝트 및 기술 코드 관련 문의
              </p>
            </div>
          </div>
        </section>

        <section className={styles.topics}>
          <h2>문의 가능한 주제</h2>
          <div className={styles.topicGrid}>
            <div className={styles.topicCategory}>
              <h3>🤖 AI/ML 기술</h3>
              <ul>
                <li>음성 인식 (STT) 기술 컨설팅</li>
                <li>딥러닝 모델 개발 및 최적화</li>
                <li>MLOps 시스템 구축</li>
                <li>AI 스타트업 기술 자문</li>
              </ul>
            </div>
            
            <div className={styles.topicCategory}>
              <h3>📝 블로그 콘텐츠</h3>
              <ul>
                <li>기술 블로그 콘텐츠 제휴</li>
                <li>게스트 포스팅 제안</li>
                <li>기술 리뷰 및 분석 의뢰</li>
                <li>콘텐츠 번역 및 공유</li>
              </ul>
            </div>
            
            <div className={styles.topicCategory}>
              <h3>💻 개발 협업</h3>
              <ul>
                <li>프로젝트 협업 문의</li>
                <li>기술 멘토링</li>
                <li>코드 리뷰 서비스</li>
                <li>기술 강연 및 워크숍</li>
              </ul>
            </div>
          </div>
        </section>

        <section className={styles.response}>
          <h2>응답 안내</h2>
          <div className={styles.responseInfo}>
            <div className={styles.responseItem}>
              <h4>📧 이메일</h4>
              <p>평일 1-2일 내 답변드리겠습니다.</p>
            </div>
            <div className={styles.responseItem}>
              <h4>🌐 소셜 미디어</h4>
              <p>가볍고 빠른 소통은 LinkedIn을 이용해주세요.</p>
            </div>
            <div className={styles.responseItem}>
              <h4>🚨 긴급 문의</h4>
              <p>이메일 제목에 [긴급]을 표기해주세요.</p>
            </div>
          </div>
        </section>

        <section className={styles.privacy}>
          <h2>개인정보 보호</h2>
          <p>
            보내주시는 모든 개인정보는 해당 문의 목적으로만 사용되며,
            제3자에게 절대 공개되지 않습니다. 자세한 내용은
            <a href="/privacy">개인정보처리방침</a>을 참고해주세요.
          </p>
        </section>

        <section className={styles.closing}>
          <p>
            여러분의 소중한 피드백과 문의를 기다리고 있습니다.
            함께 성장하는 기술 커뮤니티를 만들어가고 싶습니다.
          </p>
          <p className={styles.thanks}>
            감사합니다! 😊
          </p>
        </section>
      </div>
    </main>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => {
  const seo = {
    title: '문의',
    description: 'jiun.dev 블로그 문의 페이지 - AI/ML 기술, 블로그 콘텐츠, 개발 협업 문의',
    heroImage: ''
  }

  return <Seo title={seo.title} description={seo.description} heroImage={seo.heroImage} pathname={pathname}></Seo>
}

export default ContactPage