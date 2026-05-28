/*
 * Copyright 2025 DualverseAI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

(function () {
    const COOKIE_NAME = 'station_dashboard_theme';
    const STORAGE_KEY = 'station.dashboard.theme';
    const THEMES = ['light', 'dark', 'sci-fi'];
    const DEFAULT_THEME = 'light';
    const ONE_YEAR_SECONDS = 60 * 60 * 24 * 365;

    function normalizeTheme(theme) {
        return THEMES.includes(theme) ? theme : DEFAULT_THEME;
    }

    function readCookieTheme() {
        const cookieParts = document.cookie ? document.cookie.split(';') : [];
        for (const part of cookieParts) {
            const [name, ...valueParts] = part.trim().split('=');
            if (name === COOKIE_NAME) {
                try {
                    return normalizeTheme(decodeURIComponent(valueParts.join('=')));
                } catch (error) {
                    return DEFAULT_THEME;
                }
            }
        }
        return null;
    }

    function readStoredTheme() {
        try {
            return normalizeTheme(window.localStorage.getItem(STORAGE_KEY));
        } catch (error) {
            return null;
        }
    }

    function writeStoredTheme(theme) {
        try {
            window.localStorage.setItem(STORAGE_KEY, theme);
        } catch (error) {
            // Cookies are the primary persistence path; localStorage is only a fallback.
        }
    }

    function writeCookieTheme(theme) {
        const cookieParts = [
            `${COOKIE_NAME}=${encodeURIComponent(theme)}`,
            `Max-Age=${ONE_YEAR_SECONDS}`,
            'Path=/',
            'SameSite=Lax'
        ];
        if (window.location.protocol === 'https:') {
            cookieParts.push('Secure');
        }
        document.cookie = cookieParts.join('; ');
    }

    function getTheme() {
        return normalizeTheme(readCookieTheme() || readStoredTheme() || DEFAULT_THEME);
    }

    function setTheme(theme, options = {}) {
        const normalizedTheme = normalizeTheme(theme);
        document.documentElement.dataset.theme = normalizedTheme;
        document.documentElement.style.colorScheme = normalizedTheme === 'light' ? 'light' : 'dark';

        if (options.persist) {
            writeCookieTheme(normalizedTheme);
            writeStoredTheme(normalizedTheme);
        }

        return normalizedTheme;
    }

    window.StationTheme = {
        themes: THEMES.slice(),
        defaultTheme: DEFAULT_THEME,
        cookieName: COOKIE_NAME,
        storageKey: STORAGE_KEY,
        getTheme,
        setTheme
    };

    setTheme(getTheme());

    document.addEventListener('DOMContentLoaded', () => {
        const selector = document.getElementById('dashboard-theme-selector');
        if (!selector) return;

        selector.value = getTheme();
        selector.addEventListener('change', () => {
            selector.value = setTheme(selector.value, { persist: true });
        });
    });
})();
