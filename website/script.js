// =========================================
// PROMETHORCH - RETRO FUTURE WEBSITE
// =========================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize
    initLanguageSwitcher();
    initParallax();
    initCodeTabs();
    initNavHighlight();
    initScrollAnimations();
});

// =========================================
// LANGUAGE SWITCHER
// =========================================

let currentLang = 'ru';

function initLanguageSwitcher() {
    const langBtns = document.querySelectorAll('.lang-btn');

    langBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const lang = btn.dataset.lang;
            if (lang !== currentLang) {
                currentLang = lang;
                updateLanguage(lang);

                // Update active state
                langBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            }
        });
    });
}

function updateLanguage(lang) {
    // Update all elements with data-ru and data-en attributes
    const elements = document.querySelectorAll('[data-ru][data-en]');

    elements.forEach(el => {
        const text = el.dataset[lang];
        if (text) {
            el.textContent = text;
        }
    });

    // Update HTML lang attribute
    document.documentElement.lang = lang;
}

// =========================================
// PARALLAX EFFECT
// =========================================

function initParallax() {
    const pixelBgs = document.querySelectorAll('.pixel-bg');

    window.addEventListener('scroll', () => {
        const scrollY = window.pageYOffset;

        pixelBgs.forEach((bg, index) => {
            const speed = 0.1 + (index * 0.05);
            const rotation = (scrollY * 0.02 * (index + 1)) % 360;
            const yOffset = scrollY * speed;

            // Different effects for different backgrounds
            if (index % 3 === 0) {
                bg.style.transform = `translateY(${yOffset}px) rotate(${rotation}deg)`;
            } else if (index % 3 === 1) {
                bg.style.transform = `translateY(${-yOffset * 0.5}px) scale(${1 + scrollY * 0.0002})`;
            } else {
                bg.style.transform = `translateX(${Math.sin(scrollY * 0.01) * 20}px) translateY(${yOffset * 0.3}px)`;
            }
        });
    });
}

// =========================================
// CODE TABS
// =========================================

function initCodeTabs() {
    const tabs = document.querySelectorAll('.code-tab');
    const pythonCode = document.getElementById('python-code');
    const cppCode = document.getElementById('cpp-code');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;

            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Show/hide code
            if (tabName === 'python') {
                pythonCode.classList.remove('hidden');
                cppCode.classList.add('hidden');
            } else {
                pythonCode.classList.add('hidden');
                cppCode.classList.remove('hidden');
            }
        });
    });
}

// =========================================
// NAVIGATION HIGHLIGHT
// =========================================

function initNavHighlight() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-links a');

    window.addEventListener('scroll', () => {
        let current = '';

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (window.pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.style.color = '';
            if (link.getAttribute('href') === `#${current}`) {
                link.style.color = '#7ab87a';
            }
        });
    });

    // Smooth scroll for nav links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const target = document.querySelector(targetId);

            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// =========================================
// SCROLL ANIMATIONS
// =========================================

function initScrollAnimations() {
    const animatedElements = document.querySelectorAll(
        '.about-card, .feature-item, .timeline-item, .stat-card, .arch-box'
    );

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(el);
    });
}

// =========================================
// TYPING EFFECT (Optional for hero)
// =========================================

function typeWriter(element, text, speed = 100) {
    let i = 0;
    element.textContent = '';

    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }

    type();
}

// =========================================
// MATRIX RAIN EFFECT (Optional background)
// =========================================

function createMatrixRain(canvas) {
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const chars = 'PROMETHORCH01アイウエオカキクケコ';
    const fontSize = 14;
    const columns = canvas.width / fontSize;

    const drops = [];
    for (let i = 0; i < columns; i++) {
        drops[i] = 1;
    }

    function draw() {
        ctx.fillStyle = 'rgba(10, 10, 10, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#5c8a5c';
        ctx.font = fontSize + 'px monospace';

        for (let i = 0; i < drops.length; i++) {
            const text = chars[Math.floor(Math.random() * chars.length)];
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);

            if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
        }
    }

    setInterval(draw, 50);
}

// =========================================
// CONSOLE EASTER EGG
// =========================================

console.log(`
%c╔═══════════════════════════════════════════╗
║                                           ║
║   ██████╗ ██████╗  ██████╗ ███╗   ███╗   ║
║   ██╔══██╗██╔══██╗██╔═══██╗████╗ ████║   ║
║   ██████╔╝██████╔╝██║   ██║██╔████╔██║   ║
║   ██╔═══╝ ██╔══██╗██║   ██║██║╚██╔╝██║   ║
║   ██║     ██║  ██║╚██████╔╝██║ ╚═╝ ██║   ║
║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝   ║
║                                           ║
║   PROMETHORCH - Russian DL Framework      ║
║   Технологическая независимость           ║
║                                           ║
╚═══════════════════════════════════════════╝
`, 'color: #5c8a5c; font-family: monospace;');

console.log('%c[ СДЕЛАНО В РОССИИ ]', 'color: #c45c5c; font-weight: bold;');
