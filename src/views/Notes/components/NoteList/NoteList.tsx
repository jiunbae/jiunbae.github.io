import clsx from 'clsx'

import * as styles from './NoteList.module.scss'
import { Note } from '../Note'

type NoteListProps = {
  notes: Queries.NotesQuery['allMarkdownRemark']['nodes'];
  className?: string;
};

export const NoteList = ({ notes, className }: NoteListProps) => (
  <ul className={clsx(styles.noteList, className)}>
    {notes.length === 0 ? (
      <li className={styles.empty}>아직 등록된 노트가 없습니다.</li>
    ) : (
      notes.map(note => (
        <Note key={note.id} note={note} />
      ))
    )}
  </ul>
)
