/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ocean: {
          50: "#ecf5ff",
          100: "#d4e7ff",
          200: "#a9ceff",
          300: "#74aeff",
          400: "#4187ff",
          500: "#2563eb",
          600: "#1b49c2",
          700: "#17399b",
          800: "#182f77",
          900: "#172b61",
        },
      },
    },
  },
  plugins: [],
}

