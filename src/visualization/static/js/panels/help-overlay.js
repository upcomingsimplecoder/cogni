/**
 * panels/help-overlay.js
 * Keyboard help overlay (? key) and guided tour for first-time users.
 */

/**
 * HelpOverlay - Shows keyboard shortcut reference
 */
export class HelpOverlay {
  /**
   * Show/hide keyboard shortcut help overlay.
   * @param {HTMLElement} container - Parent element to attach to
   * @param {boolean} show - Whether to show or hide
   */
  static toggle(container, show) {
    const existingOverlay = document.getElementById('help-overlay');

    if (!show) {
      if (existingOverlay) {
        existingOverlay.remove();
      }
      return;
    }

    // Don't create duplicate overlays
    if (existingOverlay) {
      return;
    }

    // Create overlay
    const overlay = document.createElement('div');
    overlay.id = 'help-overlay';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.85);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
      animation: fadeIn 0.2s ease-out;
    `;

    // Create card
    const card = document.createElement('div');
    card.style.cssText = `
      background: #1a1a1a;
      border: 2px solid #00d4ff;
      border-radius: 12px;
      padding: 2rem;
      max-width: 600px;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 8px 32px rgba(0, 212, 255, 0.3);
      animation: slideIn 0.3s ease-out;
    `;

    // Title
    const title = document.createElement('h1');
    title.textContent = 'Keyboard Shortcuts';
    title.style.cssText = `
      color: #00d4ff;
      font-size: 1.8rem;
      margin-bottom: 1.5rem;
      text-align: center;
    `;
    card.appendChild(title);

    // Shortcut sections
    const shortcuts = [
      {
        section: 'Global Controls',
        items: [
          { key: '1-5', desc: 'Switch between lenses (Physical/Social/Cognitive/Cultural/Temporal)' },
          { key: 'Space', desc: 'Pause/Resume simulation' },
          { key: 'Esc', desc: 'Deselect agent' },
          { key: '?', desc: 'Show/hide this help' },
          { key: 'p', desc: 'Take screenshot (PNG)' },
          { key: 'e', desc: 'Export simulation data (JSON)' },
          { key: 'g', desc: 'Toggle spatial grid' }
        ]
      },
      {
        section: 'Physical Lens',
        items: [
          { key: 'Click', desc: 'Select agent to view details' }
        ]
      },
      {
        section: 'Social Lens',
        items: [
          { key: 'r', desc: 'Toggle trust relationship lines' },
          { key: 'c', desc: 'Toggle coalition convex hulls' }
        ]
      },
      {
        section: 'Cognitive Lens',
        items: [
          { key: 'd', desc: 'Toggle deliberation time heatmap' },
          { key: 'm', desc: 'Toggle Theory of Mind lines' }
        ]
      },
      {
        section: 'Cultural Lens',
        items: [
          { key: 'l', desc: 'Toggle language family bubbles' },
          { key: 't', desc: 'Toggle cultural transmission lines' }
        ]
      },
      {
        section: 'Temporal Lens',
        items: [
          { key: 't', desc: 'Toggle agent movement trails' },
          { key: 'a', desc: 'Cycle trail display mode (all/selected/none)' }
        ]
      }
    ];

    shortcuts.forEach(section => {
      const sectionDiv = document.createElement('div');
      sectionDiv.style.marginBottom = '1.5rem';

      const sectionTitle = document.createElement('h2');
      sectionTitle.textContent = section.section;
      sectionTitle.style.cssText = `
        color: #00d4ff;
        font-size: 1.2rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
      `;
      sectionDiv.appendChild(sectionTitle);

      section.items.forEach(item => {
        const shortcutRow = document.createElement('div');
        shortcutRow.style.cssText = `
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.5rem 0;
          color: #e0e0e0;
        `;

        const keySpan = document.createElement('span');
        keySpan.textContent = item.key;
        keySpan.style.cssText = `
          background: #0a0a0a;
          border: 1px solid #00d4ff;
          border-radius: 4px;
          padding: 0.25rem 0.75rem;
          font-family: 'Consolas', 'Monaco', monospace;
          font-size: 0.9rem;
          color: #00d4ff;
          min-width: 80px;
          text-align: center;
        `;

        const descSpan = document.createElement('span');
        descSpan.textContent = item.desc;
        descSpan.style.cssText = `
          flex: 1;
          margin-left: 1rem;
          font-size: 0.95rem;
        `;

        shortcutRow.appendChild(keySpan);
        shortcutRow.appendChild(descSpan);
        sectionDiv.appendChild(shortcutRow);
      });

      card.appendChild(sectionDiv);
    });

    // Dismiss instruction
    const dismissText = document.createElement('p');
    dismissText.textContent = 'Press any key or click anywhere to dismiss';
    dismissText.style.cssText = `
      text-align: center;
      color: #888;
      font-size: 0.9rem;
      margin-top: 1.5rem;
      font-style: italic;
    `;
    card.appendChild(dismissText);

    overlay.appendChild(card);
    container.appendChild(overlay);

    // Add animations
    const style = document.createElement('style');
    style.textContent = `
      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }
      @keyframes slideIn {
        from {
          transform: translateY(-20px);
          opacity: 0;
        }
        to {
          transform: translateY(0);
          opacity: 1;
        }
      }
    `;
    document.head.appendChild(style);

    // Dismiss handlers
    const dismiss = () => {
      overlay.style.animation = 'fadeOut 0.2s ease-out';
      setTimeout(() => overlay.remove(), 200);
    };

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        dismiss();
      }
    });

    document.addEventListener('keydown', dismiss, { once: true });
  }
}

/**
 * GuidedTour - First-time user onboarding tour
 */
export class GuidedTour {
  /**
   * Show guided tour overlay for first-time visitors.
   * Uses localStorage to check if tour was already shown.
   * @param {HTMLElement} container - Parent element
   */
  static maybeShow(container) {
    // Check if tour was already seen
    if (localStorage.getItem('autocog-tour-seen') === 'true') {
      return;
    }

    // Show tour
    this._showTour(container);
  }

  static _showTour(container) {
    let currentStep = 0;

    const steps = [
      {
        title: 'Welcome to AUTOCOG',
        content: 'An interactive visualization system for multi-agent artificial life simulations. Watch agents evolve, cooperate, and develop culture in real-time across five complementary perspectives.',
        icon: '🧠'
      },
      {
        title: '5 Analytical Lenses',
        content: 'Press keys <strong>1-5</strong> to switch between perspectives:<br><br><strong>1</strong> - Physical (movement, resources)<br><strong>2</strong> - Social (trust, coalitions)<br><strong>3</strong> - Cognitive (reasoning, beliefs)<br><strong>4</strong> - Cultural (language, traditions)<br><strong>5</strong> - Temporal (history, trajectories)',
        icon: '🔍'
      },
      {
        title: 'Click to Investigate',
        content: 'Click any agent to view detailed analytics in the side panels. Watch their decision-making pipeline, social relationships, cultural traits, and behavioral history unfold.',
        icon: '👆'
      },
      {
        title: 'Keyboard Shortcuts',
        content: 'Press <strong>?</strong> anytime to see all keyboard shortcuts.<br><br>Quick tips:<br><strong>Space</strong> - Pause/Resume<br><strong>p</strong> - Screenshot<br><strong>e</strong> - Export data<br><strong>Esc</strong> - Deselect agent',
        icon: '⌨️'
      }
    ];

    const createOverlay = () => {
      const overlay = document.createElement('div');
      overlay.id = 'tour-overlay';
      overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10001;
        animation: fadeIn 0.3s ease-out;
      `;

      const card = document.createElement('div');
      card.id = 'tour-card';
      card.style.cssText = `
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        border: 2px solid #00d4ff;
        border-radius: 16px;
        padding: 3rem;
        max-width: 500px;
        text-align: center;
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.4);
        animation: slideIn 0.4s ease-out;
      `;

      overlay.appendChild(card);
      return overlay;
    };

    const renderStep = (stepIndex) => {
      const step = steps[stepIndex];
      const card = document.getElementById('tour-card');
      card.innerHTML = '';

      // Icon
      const icon = document.createElement('div');
      icon.textContent = step.icon;
      icon.style.cssText = `
        font-size: 4rem;
        margin-bottom: 1.5rem;
        animation: bounce 0.6s ease-out;
      `;
      card.appendChild(icon);

      // Title
      const title = document.createElement('h1');
      title.textContent = step.title;
      title.style.cssText = `
        color: #00d4ff;
        font-size: 2rem;
        margin-bottom: 1rem;
      `;
      card.appendChild(title);

      // Content
      const content = document.createElement('p');
      content.innerHTML = step.content;
      content.style.cssText = `
        color: #e0e0e0;
        font-size: 1.1rem;
        line-height: 1.8;
        margin-bottom: 2rem;
      `;
      card.appendChild(content);

      // Progress dots
      const progressContainer = document.createElement('div');
      progressContainer.style.cssText = `
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
      `;

      steps.forEach((_, idx) => {
        const dot = document.createElement('div');
        dot.style.cssText = `
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background: ${idx === stepIndex ? '#00d4ff' : '#333'};
          transition: background 0.3s ease;
        `;
        progressContainer.appendChild(dot);
      });
      card.appendChild(progressContainer);

      // Buttons
      const buttonContainer = document.createElement('div');
      buttonContainer.style.cssText = `
        display: flex;
        gap: 1rem;
        justify-content: center;
      `;

      const skipButton = document.createElement('button');
      skipButton.textContent = 'Skip';
      skipButton.style.cssText = `
        background: transparent;
        border: 1px solid #666;
        color: #888;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
      `;
      skipButton.onmouseover = () => {
        skipButton.style.borderColor = '#00d4ff';
        skipButton.style.color = '#00d4ff';
      };
      skipButton.onmouseout = () => {
        skipButton.style.borderColor = '#666';
        skipButton.style.color = '#888';
      };
      skipButton.onclick = finishTour;

      const nextButton = document.createElement('button');
      nextButton.textContent = stepIndex === steps.length - 1 ? 'Get Started' : 'Next';
      nextButton.style.cssText = `
        background: #00d4ff;
        border: none;
        color: #0a0a0a;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
      `;
      nextButton.onmouseover = () => {
        nextButton.style.background = '#00f0ff';
        nextButton.style.transform = 'translateY(-2px)';
        nextButton.style.boxShadow = '0 4px 12px rgba(0, 212, 255, 0.5)';
      };
      nextButton.onmouseout = () => {
        nextButton.style.background = '#00d4ff';
        nextButton.style.transform = 'translateY(0)';
        nextButton.style.boxShadow = 'none';
      };
      nextButton.onclick = () => {
        if (stepIndex === steps.length - 1) {
          finishTour();
        } else {
          currentStep++;
          renderStep(currentStep);
        }
      };

      buttonContainer.appendChild(skipButton);
      buttonContainer.appendChild(nextButton);
      card.appendChild(buttonContainer);
    };

    const finishTour = () => {
      localStorage.setItem('autocog-tour-seen', 'true');
      const overlay = document.getElementById('tour-overlay');
      if (overlay) {
        overlay.style.animation = 'fadeOut 0.3s ease-out';
        setTimeout(() => overlay.remove(), 300);
      }
    };

    // Create and show overlay
    const overlay = createOverlay();
    container.appendChild(overlay);
    renderStep(0);

    // Add animations
    const style = document.createElement('style');
    style.textContent = `
      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }
      @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
      }
      @keyframes slideIn {
        from {
          transform: translateY(-30px) scale(0.95);
          opacity: 0;
        }
        to {
          transform: translateY(0) scale(1);
          opacity: 1;
        }
      }
      @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * Reset the tour state (for testing or user request)
   */
  static reset() {
    localStorage.removeItem('autocog-tour-seen');
  }
}
