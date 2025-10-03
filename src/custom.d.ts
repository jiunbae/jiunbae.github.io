declare module '*.module.css';
declare module '*.module.scss';

declare module '*.svg' {
  const content: any
  export default content
}

declare module 'wawoff2' {
  export function decompress(data: Buffer | Uint8Array): Promise<Uint8Array>
  export function compress(data: Buffer | Uint8Array): Promise<Uint8Array>
}
