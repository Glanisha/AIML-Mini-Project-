/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{ts,tsx}'],
  presets: [require('nativewind/preset')],
  // 'media' throws on web when the <html> class changes; the app is light-only anyway
  darkMode: 'class',
  theme: {
    extend: {},
  },
  plugins: [],
};
