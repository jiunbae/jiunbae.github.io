import { useCallback, useEffect, useRef, useState } from 'react';

const API_BASE = 'https://api.jiun.dev';

type CommentUser = {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
};

type Comment = {
  id: string;
  postSlug: string;
  postType: string;
  parentId: string | null;
  userId: string | null;
  user: CommentUser | null;
  anonName: string | null;
  content: string;
  isDeleted: boolean;
  createdAt: string;
  updatedAt: string | null;
  replies?: Comment[];
};

type Props = {
  postSlug: string;
  postType?: 'posts' | 'notes' | 'reviews';
};

function timeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diff = Math.floor((now - then) / 1000);

  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 2592000) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(dateStr).toLocaleDateString('ko-KR');
}

function getToken(): string | null {
  try {
    const match = document.cookie.match(/(?:^|;\s*)access_token=([^;]*)/);
    return match ? decodeURIComponent(match[1]) : null;
  } catch {
    return null;
  }
}

function getAuthHeaders(): HeadersInit {
  const token = getToken();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

async function fetchCurrentUser(): Promise<CommentUser | null> {
  const token = getToken();
  if (!token) return null;
  try {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
      credentials: 'include',
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.user || null;
  } catch {
    return null;
  }
}

function clearTokenCookie() {
  document.cookie = 'access_token=; path=/; max-age=0; SameSite=Lax';
}

export default function CommentSection({ postSlug, postType = 'posts' }: Props) {
  const [comments, setComments] = useState<Comment[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [currentUser, setCurrentUser] = useState<CommentUser | null>(null);
  const [replyTo, setReplyTo] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');

  const loadComments = useCallback(async () => {
    try {
      const res = await fetch(
        `${API_BASE}/comments?postSlug=${encodeURIComponent(postSlug)}&postType=${postType}`,
      );
      const data = await res.json();
      setComments(data.comments || []);
      setTotal(data.total || 0);
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  }, [postSlug, postType]);

  useEffect(() => {
    // Handle OAuth callback: exchange code for token
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    if (code) {
      params.delete('code');
      const cleanUrl = window.location.pathname + (params.toString() ? `?${params}` : '');
      history.replaceState(null, '', cleanUrl);

      fetch(`${API_BASE}/auth/exchange`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ code }),
      })
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data?.accessToken) {
            document.cookie = `access_token=${encodeURIComponent(data.accessToken)}; path=/; max-age=604800; SameSite=Lax`;
            fetchCurrentUser().then(setCurrentUser);
          }
        })
        .catch(() => {});
    } else {
      fetchCurrentUser().then(setCurrentUser);
    }

    loadComments();
  }, [loadComments]);

  const handleLogin = () => {
    const redirectUri = window.location.href.split('?')[0];
    window.location.href = `${API_BASE}/auth/github?redirect_uri=${encodeURIComponent(redirectUri)}`;
  };

  const handleLogout = async () => {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      });
    } catch {
      // continue with local cleanup
    }
    clearTokenCookie();
    setCurrentUser(null);
  };

  const handleSubmit = async (content: string, anonName: string, parentId?: string) => {
    const body: Record<string, string> = { postSlug, postType, content };
    if (parentId) body.parentId = parentId;
    if (!currentUser && anonName.trim()) body.anonName = anonName.trim();

    const res = await fetch(`${API_BASE}/comments`, {
      method: 'POST',
      headers: getAuthHeaders(),
      credentials: 'include',
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Failed to post comment' }));
      alert(err.error);
      return false;
    }

    setReplyTo(null);
    await loadComments();
    return true;
  };

  const handleEdit = async (commentId: string, content: string) => {
    const res = await fetch(`${API_BASE}/comments/${commentId}`, {
      method: 'PUT',
      headers: getAuthHeaders(),
      credentials: 'include',
      body: JSON.stringify({ content }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Failed to update' }));
      alert(err.error);
      return;
    }

    setEditingId(null);
    setEditContent('');
    await loadComments();
  };

  const handleDelete = async (commentId: string) => {
    if (!confirm('Delete this comment?')) return;

    const res = await fetch(`${API_BASE}/comments/${commentId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
      credentials: 'include',
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Failed to delete' }));
      alert(err.error);
      return;
    }

    await loadComments();
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.heading}>Comments {total > 0 && <span style={styles.count}>({total})</span>}</h3>

      <CommentForm
        currentUser={currentUser}
        onSubmit={(content, anonName) => handleSubmit(content, anonName)}
        onLogin={handleLogin}
        onLogout={handleLogout}
      />

      {loading ? (
        <p style={styles.loading}>Loading comments...</p>
      ) : comments.length === 0 ? (
        <p style={styles.empty}>No comments yet. Be the first!</p>
      ) : (
        <div style={styles.list}>
          {comments.map((comment) => (
            <div key={comment.id}>
              <CommentItem
                comment={comment}
                currentUser={currentUser}
                editingId={editingId}
                editContent={editContent}
                onReply={() => setReplyTo(replyTo === comment.id ? null : comment.id)}
                onEdit={() => {
                  setEditingId(comment.id);
                  setEditContent(comment.content);
                }}
                onEditCancel={() => {
                  setEditingId(null);
                  setEditContent('');
                }}
                onEditSave={() => handleEdit(comment.id, editContent)}
                onEditChange={setEditContent}
                onDelete={() => handleDelete(comment.id)}
              />

              {/* Replies */}
              {comment.replies && comment.replies.length > 0 && (
                <div style={styles.repliesContainer}>
                  {comment.replies.map((reply) => (
                    <CommentItem
                      key={reply.id}
                      comment={reply}
                      currentUser={currentUser}
                      editingId={editingId}
                      editContent={editContent}
                      isReply
                      onEdit={() => {
                        setEditingId(reply.id);
                        setEditContent(reply.content);
                      }}
                      onEditCancel={() => {
                        setEditingId(null);
                        setEditContent('');
                      }}
                      onEditSave={() => handleEdit(reply.id, editContent)}
                      onEditChange={setEditContent}
                      onDelete={() => handleDelete(reply.id)}
                    />
                  ))}
                </div>
              )}

              {/* Reply form */}
              {replyTo === comment.id && (
                <div style={styles.repliesContainer}>
                  <CommentForm
                    currentUser={currentUser}
                    onSubmit={(content, anonName) => handleSubmit(content, anonName, comment.id)}
                    placeholder="Write a reply..."
                    compact
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CommentForm({
  currentUser,
  onSubmit,
  onLogin,
  onLogout,
  placeholder = 'Write a comment...',
  compact = false,
}: {
  currentUser: CommentUser | null;
  onSubmit: (content: string, anonName: string) => Promise<boolean>;
  onLogin?: () => void;
  onLogout?: () => void;
  placeholder?: string;
  compact?: boolean;
}) {
  const [content, setContent] = useState('');
  const [anonName, setAnonName] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!content.trim()) return;
    setSubmitting(true);
    const ok = await onSubmit(content, anonName);
    if (ok) {
      setContent('');
      setAnonName('');
    }
    setSubmitting(false);
  };

  return (
    <form onSubmit={handleSubmit} style={{ ...styles.form, ...(compact ? styles.formCompact : {}) }}>
      {!currentUser ? (
        <div style={styles.formHeader}>
          <input
            type="text"
            value={anonName}
            onChange={(e) => setAnonName(e.target.value)}
            placeholder="Name"
            maxLength={50}
            required
            style={styles.input}
          />
          {onLogin && (
            <button type="button" onClick={onLogin} style={styles.loginBtn}>
              Login with GitHub &rarr;
            </button>
          )}
        </div>
      ) : (
        <div style={styles.formHeader}>
          <span style={styles.userInfo}>
            {currentUser.avatarUrl && (
              <img src={currentUser.avatarUrl} alt="" style={styles.avatar} />
            )}
            <span style={styles.loggedInAs}>
              {currentUser.displayName || currentUser.username}
            </span>
          </span>
          {onLogout && (
            <button type="button" onClick={onLogout} style={styles.loginBtn}>
              Logout
            </button>
          )}
        </div>
      )}
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder={placeholder}
        maxLength={3000}
        required
        rows={compact ? 2 : 3}
        style={styles.textarea}
      />
      <div style={styles.formActions}>
        <button type="submit" disabled={submitting || !content.trim()} style={styles.submitBtn}>
          {submitting ? 'Posting...' : 'Post'}
        </button>
      </div>
    </form>
  );
}

function CommentItem({
  comment,
  currentUser,
  editingId,
  editContent,
  isReply = false,
  onReply,
  onEdit,
  onEditCancel,
  onEditSave,
  onEditChange,
  onDelete,
}: {
  comment: Comment;
  currentUser: CommentUser | null;
  editingId: string | null;
  editContent: string;
  isReply?: boolean;
  onReply?: () => void;
  onEdit: () => void;
  onEditCancel: () => void;
  onEditSave: () => void;
  onEditChange: (v: string) => void;
  onDelete: () => void;
}) {
  const isOwner = currentUser && comment.userId === currentUser.id;
  const isEditing = editingId === comment.id;
  const displayName = comment.user?.displayName || comment.user?.username || comment.anonName || 'Anonymous';

  if (comment.isDeleted) {
    return (
      <div style={{ ...styles.comment, ...(isReply ? styles.commentReply : {}) }}>
        <p style={styles.deleted}>This comment has been deleted.</p>
      </div>
    );
  }

  return (
    <div style={{ ...styles.comment, ...(isReply ? styles.commentReply : {}) }}>
      <div style={styles.commentHeader}>
        <span style={styles.author}>
          {displayName}
          {comment.user && <span style={styles.badge}>user</span>}
        </span>
        <span style={styles.time}>
          {timeAgo(comment.createdAt)}
          {comment.updatedAt && ' (edited)'}
        </span>
      </div>

      {isEditing ? (
        <div style={styles.editArea}>
          <textarea
            value={editContent}
            onChange={(e) => onEditChange(e.target.value)}
            rows={3}
            maxLength={3000}
            style={styles.textarea}
          />
          <div style={styles.editActions}>
            <button onClick={onEditCancel} style={styles.linkBtn}>Cancel</button>
            <button onClick={onEditSave} style={styles.submitBtn}>Save</button>
          </div>
        </div>
      ) : (
        <p style={styles.content}>{comment.content}</p>
      )}

      {!isEditing && (
        <div style={styles.actions}>
          {onReply && !isReply && (
            <button onClick={onReply} style={styles.linkBtn}>Reply</button>
          )}
          {isOwner && (
            <>
              <button onClick={onEdit} style={styles.linkBtn}>Edit</button>
              <button onClick={onDelete} style={{ ...styles.linkBtn, ...styles.deleteBtn }}>Delete</button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    fontFamily: 'inherit',
  },
  heading: {
    fontSize: '1.25rem',
    fontWeight: 600,
    color: 'var(--gray-1)',
    marginBottom: '1.5rem',
  },
  count: {
    fontWeight: 400,
    color: 'var(--gray-4)',
    fontSize: '1rem',
  },
  loading: {
    color: 'var(--gray-4)',
    textAlign: 'center',
    padding: '2rem 0',
  },
  empty: {
    color: 'var(--gray-4)',
    textAlign: 'center',
    padding: '2rem 0',
    fontStyle: 'italic',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    marginTop: '1.5rem',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  formCompact: {
    gap: '0.5rem',
  },
  input: {
    padding: '0.5rem 0.75rem',
    border: '1px solid var(--gray-5)',
    borderRadius: '0.375rem',
    background: 'var(--article-background)',
    color: 'var(--gray-1)',
    fontSize: '0.875rem',
    maxWidth: '200px',
    fontFamily: 'inherit',
  },
  textarea: {
    padding: '0.75rem',
    border: '1px solid var(--gray-5)',
    borderRadius: '0.375rem',
    background: 'var(--article-background)',
    color: 'var(--gray-1)',
    fontSize: '0.9375rem',
    lineHeight: 1.5,
    resize: 'vertical' as const,
    fontFamily: 'inherit',
    minHeight: '60px',
  },
  formHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '0.75rem',
  },
  userInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  avatar: {
    width: '24px',
    height: '24px',
    borderRadius: '50%',
  },
  loginBtn: {
    background: 'none',
    border: 'none',
    color: 'var(--gray-4)',
    fontSize: '0.8125rem',
    cursor: 'pointer',
    padding: 0,
    fontFamily: 'inherit',
    whiteSpace: 'nowrap' as const,
  },
  formActions: {
    display: 'flex',
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  loggedInAs: {
    fontSize: '0.8125rem',
    color: 'var(--gray-4)',
    fontWeight: 500,
  },
  submitBtn: {
    padding: '0.5rem 1.25rem',
    background: 'var(--primary-c1)',
    color: '#fff',
    border: 'none',
    borderRadius: '0.375rem',
    fontSize: '0.875rem',
    fontWeight: 500,
    cursor: 'pointer',
    marginLeft: 'auto',
    fontFamily: 'inherit',
  },
  comment: {
    padding: '1rem 0',
    borderBottom: '1px solid var(--gray-5)',
  },
  commentReply: {
    borderBottom: 'none',
    paddingTop: '0.75rem',
    paddingBottom: '0.5rem',
  },
  commentHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.375rem',
  },
  author: {
    fontWeight: 600,
    fontSize: '0.9375rem',
    color: 'var(--gray-1)',
    display: 'flex',
    alignItems: 'center',
    gap: '0.375rem',
  },
  badge: {
    fontSize: '0.6875rem',
    padding: '0.1rem 0.375rem',
    borderRadius: '0.25rem',
    background: 'var(--primary-c1)',
    color: '#fff',
    fontWeight: 500,
  },
  time: {
    fontSize: '0.8125rem',
    color: 'var(--gray-4)',
  },
  content: {
    fontSize: '0.9375rem',
    lineHeight: 1.7,
    color: 'var(--gray-2)',
    margin: '0.25rem 0 0.5rem',
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
  },
  deleted: {
    fontSize: '0.875rem',
    color: 'var(--gray-4)',
    fontStyle: 'italic',
    margin: '0.25rem 0',
  },
  actions: {
    display: 'flex',
    gap: '0.75rem',
  },
  linkBtn: {
    background: 'none',
    border: 'none',
    color: 'var(--gray-4)',
    fontSize: '0.8125rem',
    cursor: 'pointer',
    padding: 0,
    fontFamily: 'inherit',
  },
  deleteBtn: {
    color: '#c0392b',
  },
  editArea: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    marginTop: '0.375rem',
  },
  editActions: {
    display: 'flex',
    gap: '0.5rem',
    justifyContent: 'flex-end',
  },
  repliesContainer: {
    marginLeft: '1.5rem',
    borderLeft: '2px solid var(--gray-5)',
    paddingLeft: '1rem',
  },
};
