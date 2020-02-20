module.exports = {
    root: true,
    parser: '@typescript-eslint/parser',
    plugins: [
        '@typescript-eslint', "react", 'react-hooks', 'eslint-comments'
    ],
    extends: [
        "react-app",
        'eslint:recommended',
        'plugin:@typescript-eslint/eslint-recommended',
        'plugin:@typescript-eslint/recommended',
        "plugin:react/recommended",
        'standard-with-typescript',
    ],
    parserOptions: {
        project: "./tsconfig.json",
        sourceType:  'module',  // Allows for the use of imports
    },
    rules: {
        "react-hooks/rules-of-hooks": "error",
        "react-hooks/exhaustive-deps": "warn",
        "@typescript-eslint/interface-name-prefix": ["error", {"prefixWithI": "always"}],
        "@typescript-eslint/indent": ["error", 4, { 'SwitchCase': 1 }],
        "jsx-quotes": ["error", "prefer-single"],
        '@typescript-eslint/no-unused-vars': ['error', {
            'vars': 'all',
            'args': 'none',
            'ignoreRestSiblings': true,
        }],
        "@typescript-eslint/strict-boolean-expressions": 0,
    },
    settings:  {
        react:  {
            version:  'detect',  // Tells eslint-plugin-react to automatically detect the version of React to use
        },
    }
};
