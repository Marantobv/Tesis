/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        'verde': '#C2C6A7',
        'morado': '#270722',
        'celeste': '#9AC2C5',
        'amarillo': '#DBCF96',
        'naranja': '#ECCE8E'
      },
      fontFamily: {
        primary: ['Righteous', "sans-serif"],
        secondary: ['Roboto', "sans-serif"],
        tertiary: ['Kanit', "sans-serif"],

      }
    },
    backgroundSize:{
      btn: '200% auto'
    },
    backgroundPosition:{
      btnhover: 'right center'
    }
  },
  plugins: [],
}
