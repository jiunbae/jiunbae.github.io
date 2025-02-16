// @ts-check

import eslint from '@eslint/js'
import * as importPlugin from 'eslint-plugin-import'
import jsxA11yPlugin from 'eslint-plugin-jsx-a11y'
import reactPlugin from 'eslint-plugin-react'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  eslint.configs.recommended,
  tseslint.configs.recommended,
  {
    ignores: ['public/**', '.cache/**', 'node_modules/**']
  },
  {
    files: ['**/*.{ts,tsx,d.ts}'],
    plugins: {
      ['@typescript-eslint']: tseslint.plugin,
      ['import']: importPlugin,
      ['react']: reactPlugin,
      ['jsx-a11y']: jsxA11yPlugin,
    },
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node
      },
      parserOptions: {
        project: './tsconfig.json',
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    settings: {
      react: { 
        version: 'detect' 
      },
      'import/resolver': {
        typescript: { 
          project: './tsconfig.json' 
        },
        // alias: {
        //   map: [['@', './src']],
        //   extensions: ['.js', '.jsx', '.ts', '.tsx']
        // }
      }
    },
    rules: {
      quotes: ['error', 'single'],
      semi: ['error', 'never'],
      'comma-dangle': ['error', 'never'],
      'react/react-in-jsx-scope': 'off',
      'react/no-unescaped-entities': 'off',
      'react/jsx-filename-extension': ['warn', { 
        extensions: ['.tsx', '.jsx'] 
      }],
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': ['error', {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
        ignoreRestSiblings: true
      }]
    }
  }
)

