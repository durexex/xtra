import React from 'react';
import './RadioGroup.css';

const RadioGroup = ({ options, selectedOption, onChange }) => {
  return (
    <div className="radio-group">
      {options.map((option, index) => (
        <label key={index} className="radio-label">
          <input
            type="radio"
            name="radio-group"
            value={option}
            checked={selectedOption === option}
            onChange={(e) => onChange(e.target.value)}
            className="radio-input"
          />
          {option}
        </label>
      ))}
    </div>
  );
};

export default RadioGroup;