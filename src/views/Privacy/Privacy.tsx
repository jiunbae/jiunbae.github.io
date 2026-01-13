import type { HeadProps } from 'gatsby'
import { Seo } from '@/components'

import * as styles from './Privacy.module.scss'

const PrivacyPage = () => {
  return (
    <main className={styles.page}>
      <div className={styles.content}>
        <h1>개인정보처리방침</h1>
        
        <section>
          <h2>1. 개인정보의 수집 및 이용 목적</h2>
          <p>본 블로그는 다음의 목적을 위해 개인정보를 수집 및 이용합니다:</p>
          <ul>
            <li>블로그 콘텐츠 제공 및 서비스 운영</li>
            <li>댓글 및 소통 기능 제공</li>
            <li>서비스 개선 및 사용자 경험 향상</li>
            <li>문의사항 처리 및 고객 지원</li>
          </ul>
        </section>

        <section>
          <h2>2. 수집하는 개인정보 항목</h2>
          <p>본 블로그는 다음과 같은 최소한의 개인정보를 수집합니다:</p>
          <ul>
            <li>수집항목: 이메일 주소 (문의 및 댓글 작성 시)</li>
            <li>수집방법: 홈페이지 문의 양식, 댓글 작성</li>
            <li>보유기간: 서비스 운영 기간 또는 동의 철회 시까지</li>
          </ul>
        </section>

        <section>
          <h2>3. 개인정보의 제3자 제공</h2>
          <p>본 블로그는 이용자의 동의가 있는 경우나 법률에 특별한 규정이 있는 경우를 제외하고는 
          개인정보를 제3자에게 제공하지 않습니다.</p>
        </section>

        <section>
          <h2>4. 개인정보의 처리위탁</h2>
          <p>본 블로그는 서비스 향상을 위해 다음과 같이 개인정보 처리를 위탁하고 있으며, 
          관련 법규에 따라 위탁계약 시 개인정보가 안전하게 관리되도록 필요한 조치를 하고 있습니다.</p>
          <ul>
            <li>수탁업체: Google (Google Analytics)</li>
            <li>위탁업무: 웹사이트 분석 및 사용자 행동 패턴 분석</li>
          </ul>
        </section>

        <section>
          <h2>5. 이용자 권리와 행사 방법</h2>
          <p>이용자는 언제든지 다음의 권리를 행사할 수 있습니다:</p>
          <ul>
            <li>개인정보 열람 요구</li>
            <li>오류 등이 있을 경우 정정 요구</li>
            <li>삭제 요구</li>
            <li>처리 정지 요구</li>
          </ul>
        </section>

        <section>
          <h2>6. 개인정보의 파기</h2>
          <p>이용자의 개인정보는 개인정보의 수집 및 이용목적이 달성되면 지체 없이 파기됩니다. 
          파기 절차 및 방법은 다음과 같습니다:</p>
          <ul>
            <li>파기 절차: 이용자가 입력한 정보는 목적 달성 후 별도의 DB에 옮겨져 
            내부 방침 및 기타 관련 법령에 따라 일정기간 저장 후 파기됩니다.</li>
            <li>파기 방법: 종이에 출력된 개인정보는 분쇄기로 분쇄하거나 소각하고, 
            전자적 파일 형태의 정보는 기록을 재생할 수 없는 기술적 방법을 사용하여 삭제합니다.</li>
          </ul>
        </section>

        <section>
          <h2>7. 개인정보 보호를 위한 기술적/관리적 조치</h2>
          <p>본 블로그는 이용자의 개인정보를 안전하게 관리하기 위하여 다음과 같은 기술적/관리적 조치를 취하고 있습니다:</p>
          <ul>
            <li>기술적 조치: 비밀번호 암호화, 해킹 방지를 위한 보안시스템 구축</li>
            <li>관리적 조치: 개인정보처리 직원의 최소화 및 정기적인 교육</li>
          </ul>
        </section>

        <section>
          <h2>8. 쿠키(Cookie)의 운영 및 거부</h2>
          <p>본 블로그는 서비스 운영 및 개선을 위해 쿠키를 사용합니다. 쿠키는 웹사이트를 운영하는데 이용되는 
          서버가 사용자의 브라우저에 보내는 소량의 정보이며, 이용자들의 컴퓨터에 저장됩니다.</p>
          <p>이용자는 쿠키 설정을 변경하여 쿠키에 대한 수신을 거부할 수 있으며, 
          이 경우 서비스의 일부 기능이 제한될 수 있습니다.</p>
        </section>

        <section>
          <h2>9. 개인정보 보호책임자</h2>
          <p>본 블로그는 이용자의 개인정보를 보호하고 개인정보와 관련한 불만을 처리하기 위하여 
          아래와 같이 개인정보 보호책임자를 지정하고 있습니다.</p>
          <ul>
            <li>이름: 배지운</li>
            <li>이메일: contact@jiun.dev</li>
            <li>연락처: 별도의 연락처는 공개하지 않습니다.</li>
          </ul>
        </section>

        <section>
          <h2>10. 개인정보처리방침의 변경</h2>
          <p>본 개인정보처리방침은 시행일로부터 적용되며, 법령 및 방침에 따라 변경될 수 있습니다. 
          개인정보처리방침이 변경되는 경우에는 변경사항을 웹사이트에 공지합니다.</p>
          <p><strong>시행일자: 2024년 1월 1일</strong></p>
          <p><strong>최종 개정일자: 2024년 12월 1일</strong></p>
        </section>
      </div>
    </main>
  )
}

export const Head = ({ location: { pathname } }: HeadProps) => {
  const seo = {
    title: '개인정보처리방침',
    description: 'jiun.dev 블로그의 개인정보처리방침 안내 페이지',
    heroImage: ''
  }

  return <Seo title={seo.title} description={seo.description} heroImage={seo.heroImage} pathname={pathname}></Seo>
}

export default PrivacyPage
