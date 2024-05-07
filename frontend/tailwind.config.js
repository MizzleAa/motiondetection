/** @type {import('tailwindcss').Config} */

const colors = require('tailwindcss/colors')
const defaultTheme = require('tailwindcss/defaultTheme')
const defaultColors = require('tailwindcss/colors')

module.exports = {
  darkMode: 'class',
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx}",
    "./src/components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    colors: {
      transparent: 'transparent',
      current: 'currentColor',
      slate:colors.slate,
      gray:colors.gray,
      zinc:colors.zinc,
      neutral:colors.neutral,
      stone:colors.stone,
      red:colors.red,
      orange:colors.orange,
      amber:colors.amber,
      yellow:colors.yellow,
      lime:colors.lime,
      green:colors.green,
      emerald:colors.emerald,
      teal:colors.teal,
      cyan:colors.cyan,
      sky:colors.sky,
      blue:colors.blue,
      violet:colors.violet,
      purple:colors.purple,
      fuchsia:colors.fuchsia,
      pink:colors.pink,
      rose:colors.rose,
      ...defaultColors.colors,
    },
    screens: {
      '2xs': {'min': '360px', 'max': '479px'},
      // => @media (min-width: 360px and max-width: 479px) { ... }

      'xs': {'min': '480px', 'max': '639px'},
      // => @media (min-width: 480px and max-width: 639px) { ... }

      'sm': {'min': '640px', 'max': '767px'},
      // => @media (min-width: 640px and max-width: 767px) { ... }

      'md': {'min': '768px', 'max': '1023px'},
      // => @media (min-width: 768px and max-width: 1023px) { ... }

      'lg': {'min': '1024px', 'max': '1279px'},
      // => @media (min-width: 1024px and max-width: 1279px) { ... }

      'xl': {'min': '1280px', 'max': '1535px'},
      // => @media (min-width: 1280px and max-width: 1535px) { ... }

      '2xl': {'min': '1536px', 'max': '1919px'},
      // k => @media (min-width: 1536px and max-width: 1919px) { ... }
      
      '3xl': {'min': '1920px'},
      // 2k => @media (min-width: 1921px ) { ... }

      'tablet': '640px',
      // => @media (min-width: 640px) { ... }
      'laptop': '1024px',
      // => @media (min-width: 1024px) { ... }
      'desktop': '1280px',
      // => @media (min-width: 1280px) { ... }
      ...defaultTheme.screens,
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('@tailwindcss/forms'),
    require('@tailwindcss/line-clamp'),
    require('@tailwindcss/aspect-ratio'),
    // require("flowbite/plugin")
  ],
}