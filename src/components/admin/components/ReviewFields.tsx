interface ReviewFieldsProps {
  rating: number;
  onChange: (rating: number) => void;
}

function renderStars(rating: number): string {
  const fullStars = Math.floor(rating);
  const halfStar = rating % 1 >= 0.5 ? 1 : 0;
  const emptyStars = 5 - fullStars - halfStar;
  return (
    "\u2605".repeat(fullStars) +
    (halfStar ? "\u00BD" : "") +
    "\u2606".repeat(emptyStars)
  );
}

export default function ReviewFields({ rating, onChange }: ReviewFieldsProps) {
  return (
    <div className="form-group">
      <label htmlFor="rf-rating">
        Rating: {renderStars(rating)} ({rating}/5)
      </label>
      <input
        id="rf-rating"
        type="number"
        min={0}
        max={5}
        step={0.5}
        value={rating}
        onChange={(e) => {
          const value = parseFloat(e.target.value);
          if (!isNaN(value) && value >= 0 && value <= 5) {
            onChange(value);
          }
        }}
      />
    </div>
  );
}
