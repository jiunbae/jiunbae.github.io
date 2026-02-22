/**
 * 암호화 유틸리티
 * Web Crypto API를 사용한 AES-GCM 암호화
 *
 * SECURITY NOTE:
 * - 브라우저의 Web Crypto API 사용
 * - AES-GCM (256-bit) 암호화 알고리즘
 * - 매 암호화마다 새로운 IV (Initialization Vector) 생성
 * - 암호화 키는 사용자의 브라우저 환경에서 파생
 */

const ALGORITHM = 'AES-GCM'
const KEY_LENGTH = 256
const IV_LENGTH = 12 // 96 bits for GCM
type StrictUint8Array = Uint8Array<ArrayBuffer>

/**
 * 텍스트를 ArrayBuffer로 변환
 */
const textToBuffer = (text: string): StrictUint8Array => {
  return new TextEncoder().encode(text) as StrictUint8Array
}

/**
 * ArrayBuffer를 텍스트로 변환
 */
const bufferToText = (buffer: ArrayBuffer): string => {
  return new TextDecoder().decode(buffer)
}

/**
 * ArrayBuffer를 Base64로 인코딩
 */
const bufferToBase64 = (buffer: Uint8Array): string => {
  let binary = ''
  const len = buffer.byteLength
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(buffer[i])
  }
  return btoa(binary)
}

/**
 * Base64를 ArrayBuffer로 디코딩
 */
const base64ToBuffer = (base64: string): StrictUint8Array => {
  const binary = atob(base64)
  const len = binary.length
  const buffer = new Uint8Array(new ArrayBuffer(len))
  for (let i = 0; i < len; i++) {
    buffer[i] = binary.charCodeAt(i)
  }
  return buffer as StrictUint8Array
}

/**
 * 암호화 키 생성
 * 브라우저 지문(fingerprint)을 기반으로 키 생성
 */
const deriveKey = async (passphrase: string): Promise<CryptoKey> => {
  const encoder = new TextEncoder()
  const passphraseKey = await crypto.subtle.importKey(
    'raw',
    encoder.encode(passphrase),
    'PBKDF2',
    false,
    ['deriveBits', 'deriveKey']
  )

  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: encoder.encode('jiunbae-blog-salt'), // 고정 salt (앱별로 다르게)
      iterations: 100000,
      hash: 'SHA-256',
    },
    passphraseKey,
    { name: ALGORITHM, length: KEY_LENGTH },
    false,
    ['encrypt', 'decrypt']
  )
}

/**
 * 브라우저 지문 생성
 * 사용자 환경 기반의 간단한 키 생성
 */
const getBrowserFingerprint = (): string => {
  if (typeof window === 'undefined') {
    return 'server-side-key'
  }

  // 브라우저 환경 정보 결합
  const fingerprint = [
    navigator.userAgent,
    navigator.language,
    new Date().getTimezoneOffset(),
    screen.width + 'x' + screen.height,
    'jiunbae-blog-key', // 추가 엔트로피
  ].join('|')

  return fingerprint
}

/**
 * 텍스트 암호화
 * @param plaintext 암호화할 평문
 * @returns Base64로 인코딩된 암호문 (IV + 암호문)
 */
export const encrypt = async (plaintext: string): Promise<string> => {
  try {
    const fingerprint = getBrowserFingerprint()
    const key = await deriveKey(fingerprint)

    // 랜덤 IV 생성
    const iv = crypto.getRandomValues(new Uint8Array(IV_LENGTH)) as StrictUint8Array

    // 암호화
    const encrypted = await crypto.subtle.encrypt(
      {
        name: ALGORITHM,
        iv: iv,
      },
      key,
      textToBuffer(plaintext)
    )

    // IV + 암호문을 결합하여 Base64로 인코딩
    const encryptedBuffer = new Uint8Array(encrypted)
    const combined = new Uint8Array(iv.length + encryptedBuffer.length)
    combined.set(iv, 0)
    combined.set(encryptedBuffer, iv.length)

    return bufferToBase64(combined)
  } catch (error) {
    console.error('Encryption failed:', error)
    throw new Error('암호화에 실패했습니다')
  }
}

/**
 * 텍스트 복호화
 * @param ciphertext Base64로 인코딩된 암호문
 * @returns 복호화된 평문
 */
export const decrypt = async (ciphertext: string): Promise<string> => {
  try {
    const fingerprint = getBrowserFingerprint()
    const key = await deriveKey(fingerprint)

    // Base64 디코딩
    const combined = base64ToBuffer(ciphertext)

    // IV와 암호문 분리
    const iv = combined.slice(0, IV_LENGTH)
    const encrypted = combined.slice(IV_LENGTH)

    // 복호화
    const decrypted = await crypto.subtle.decrypt(
      {
        name: ALGORITHM,
        iv: iv,
      },
      key,
      encrypted
    )

    return bufferToText(decrypted)
  } catch (error) {
    console.error('Decryption failed:', error)
    throw new Error('복호화에 실패했습니다. 토큰이 손상되었을 수 있습니다.')
  }
}

/**
 * 암호화된 데이터 검증
 * @param ciphertext 검증할 암호문
 * @returns 유효한 암호문인지 여부
 */
export const isValidEncryptedData = (ciphertext: string): boolean => {
  try {
    const buffer = base64ToBuffer(ciphertext)
    return buffer.length > IV_LENGTH
  } catch {
    return false
  }
}
