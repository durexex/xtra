import React from 'react';
import './Modal.css';

const Modal = ({ isOpen, onClose, title, children, contentClassName }) => {
  if (!isOpen) {
    return null;
  }

  const contentClasses = ['modal-content', contentClassName].filter(Boolean).join(' ');

  return (
    <div className="modal">
      <div className={contentClasses}>
        <span className="close" onClick={onClose}>&times;</span>
        <h2>{title}</h2>
        {children}
      </div>
    </div>
  );
};

export default Modal;
