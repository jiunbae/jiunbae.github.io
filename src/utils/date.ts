/** Short date format: 25.03.02 */
export const fmtDate = (d: Date) => {
  const y = String(d.getFullYear()).slice(2);
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}.${m}.${day}`;
};

/** Korean date format: 2025년 03월 02일 */
export const fmtDateKr = (d: Date) =>
  `${d.getFullYear()}년 ${String(d.getMonth() + 1).padStart(2, '0')}월 ${String(d.getDate()).padStart(2, '0')}일`;
