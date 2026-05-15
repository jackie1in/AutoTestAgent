_PAGE_INFO_JS = """() => {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const pw = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth || 0);
    const ph = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight || 0);
    const sx = window.scrollX || window.pageXOffset || document.documentElement.scrollLeft || 0;
    const sy = window.scrollY || window.pageYOffset || document.documentElement.scrollTop || 0;
    const pb = Math.max(0, ph - (vh + sy));
    return JSON.stringify({
        viewport_width: vw, viewport_height: vh,
        page_width: pw, page_height: ph,
        scroll_x: sx, scroll_y: sy,
        pixels_above: sy, pixels_below: pb,
        pages_above: vh > 0 ? sy / vh : 0,
        pages_below: vh > 0 ? pb / vh : 0,
        total_pages: vh > 0 ? ph / vh : 0,
        current_page_position: sy / Math.max(1, ph - vh)
    });
}"""

_CLICK_ELEMENT_JS = """(el) => {
    if (typeof el.scrollIntoViewIfNeeded === 'function') {
        el.scrollIntoViewIfNeeded(true);
    } else {
        el.scrollIntoView({behavior: 'auto', block: 'center', inline: 'nearest'});
    }

    const rect = el.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;

    const hitTarget = document.elementFromPoint(x, y);
    const target = (hitTarget instanceof HTMLElement && el.contains(hitTarget))
        ? hitTarget : el;

    const pointerOpts = {
        bubbles: true, cancelable: true,
        clientX: x, clientY: y, pointerType: 'mouse'
    };
    const mouseOpts = {
        bubbles: true, cancelable: true,
        clientX: x, clientY: y, button: 0
    };

    target.dispatchEvent(new PointerEvent('pointerover', pointerOpts));
    target.dispatchEvent(new PointerEvent('pointerenter', {...pointerOpts, bubbles: false}));
    target.dispatchEvent(new MouseEvent('mouseover', mouseOpts));
    target.dispatchEvent(new MouseEvent('mouseenter', {...mouseOpts, bubbles: false}));

    target.dispatchEvent(new PointerEvent('pointerdown', pointerOpts));
    target.dispatchEvent(new MouseEvent('mousedown', mouseOpts));

    el.focus({preventScroll: true});

    target.dispatchEvent(new PointerEvent('pointerup', pointerOpts));
    target.dispatchEvent(new MouseEvent('mouseup', mouseOpts));

    target.click();

    const isAnchor = el.tagName === 'A';
    const isBlank = isAnchor && el.target === '_blank';
    return JSON.stringify({isAnchor, isBlank});
}"""

_INPUT_TEXT_JS = """(el, text) => {
    const isContentEditable = el.isContentEditable;
    const isInput = el.tagName === 'INPUT';
    const isTextArea = el.tagName === 'TEXTAREA';

    if (isContentEditable) {
        if (el.dispatchEvent(new InputEvent('beforeinput', {
            bubbles: true, cancelable: true, inputType: 'deleteContent'
        }))) {
            el.innerText = '';
            el.dispatchEvent(new InputEvent('input', {
                bubbles: true, inputType: 'deleteContent'
            }));
        }
        if (el.dispatchEvent(new InputEvent('beforeinput', {
            bubbles: true, cancelable: true, inputType: 'insertText', data: text
        }))) {
            el.innerText = text;
            el.dispatchEvent(new InputEvent('input', {
                bubbles: true, inputType: 'insertText', data: text
            }));
        }

        const planAOk = el.innerText.trim() === text.trim();
        if (!planAOk) {
            el.focus();
            const doc = el.ownerDocument;
            const sel = (doc.defaultView || window).getSelection();
            const range = doc.createRange();
            range.selectNodeContents(el);
            sel.removeAllRanges();
            sel.addRange(range);
            doc.execCommand('delete', false);
            doc.execCommand('insertText', false, text);
        }
        el.dispatchEvent(new Event('change', {bubbles: true}));
        el.blur();
        return JSON.stringify({success: true, method: planAOk ? 'synthetic' : 'execCommand'});
    }

    if (isInput || isTextArea) {
        const proto = Object.getPrototypeOf(el);
        const setter = Object.getOwnPropertyDescriptor(proto, 'value');
        if (setter && setter.set) {
            setter.set.call(el, text);
        } else {
            el.value = text;
        }
        el.dispatchEvent(new Event('input', {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        el.blur();
        return JSON.stringify({success: true, method: 'nativeValueSetter'});
    }

    return JSON.stringify({success: false, method: 'unsupported'});
}"""

_SCROLL_VERTICAL_JS = """(el, dy) => {
    if (el) {
        let cur = el;
        let attempts = 0;
        while (cur && attempts < 10) {
            const cs = window.getComputedStyle(cur);
            const hasScrollY = /(auto|scroll|overlay)/.test(cs.overflowY);
            const canScroll = cur.scrollHeight > cur.clientHeight;
            if (hasScrollY && canScroll) {
                const before = cur.scrollTop;
                const max = cur.scrollHeight - cur.clientHeight;
                let amt = dy / 3;
                if (amt > 0) amt = Math.min(amt, max - before);
                else amt = Math.max(amt, -before);
                cur.scrollTop = before + amt;
                const delta = cur.scrollTop - before;
                if (Math.abs(delta) > 0.5) {
                    return JSON.stringify({
                        success: true,
                        message: 'Scrolled container (' + cur.tagName + ') by ' + delta + 'px'
                    });
                }
            }
            if (cur === document.body || cur === document.documentElement) break;
            cur = cur.parentElement;
            attempts++;
        }
        return JSON.stringify({
            success: false,
            message: 'No scrollable container found for element (' + el.tagName + ')'
        });
    }

    const before = window.scrollY;
    const maxS = document.documentElement.scrollHeight - window.innerHeight;
    window.scrollBy(0, dy);
    const after = window.scrollY;
    const scrolled = after - before;
    if (Math.abs(scrolled) < 1) {
        const msg = dy > 0
            ? 'Already at the bottom of the page'
            : 'Already at the top of the page';
        return JSON.stringify({success: false, message: msg});
    }
    const atEnd = (dy > 0 && after >= maxS - 1) || (dy < 0 && after <= 1);
    const edge = dy > 0 ? 'bottom' : 'top';
    const msg = atEnd
        ? 'Scrolled page by ' + scrolled + 'px. Reached the ' + edge + '.'
        : 'Scrolled page by ' + scrolled + 'px.';
    return JSON.stringify({success: true, message: msg});
}"""

_SCROLL_HORIZONTAL_JS = """(el, dx) => {
    if (el) {
        let cur = el;
        let attempts = 0;
        while (cur && attempts < 10) {
            const cs = window.getComputedStyle(cur);
            const hasScrollX = /(auto|scroll|overlay)/.test(cs.overflowX);
            const canScroll = cur.scrollWidth > cur.clientWidth;
            if (hasScrollX && canScroll) {
                const before = cur.scrollLeft;
                const max = cur.scrollWidth - cur.clientWidth;
                let amt = dx / 3;
                if (amt > 0) amt = Math.min(amt, max - before);
                else amt = Math.max(amt, -before);
                cur.scrollLeft = before + amt;
                const delta = cur.scrollLeft - before;
                if (Math.abs(delta) > 0.5) {
                    return JSON.stringify({
                        success: true,
                        message: 'Scrolled container (' + cur.tagName + ') horizontally by ' + delta + 'px'
                    });
                }
            }
            if (cur === document.body || cur === document.documentElement) break;
            cur = cur.parentElement;
            attempts++;
        }
        return JSON.stringify({
            success: false,
            message: 'No horizontally scrollable container for element (' + el.tagName + ')'
        });
    }

    const before = window.scrollX;
    const maxS = document.documentElement.scrollWidth - window.innerWidth;
    window.scrollBy(dx, 0);
    const after = window.scrollX;
    const scrolled = after - before;
    if (Math.abs(scrolled) < 1) {
        const msg = dx > 0
            ? 'Already at the right edge of the page'
            : 'Already at the left edge of the page';
        return JSON.stringify({success: false, message: msg});
    }
    const atEnd = (dx > 0 && after >= maxS - 1) || (dx < 0 && after <= 1);
    const edge = dx > 0 ? 'right edge' : 'left edge';
    const msg = atEnd
        ? 'Scrolled page horizontally by ' + scrolled + 'px. Reached the ' + edge + '.'
        : 'Scrolled page horizontally by ' + scrolled + 'px.';
    return JSON.stringify({success: true, message: msg});
}"""

_PATCH_REACT_JS = """() => {
    const roots = document.querySelectorAll(
        '[data-reactroot], [data-reactid], [data-react-checksum], ' +
        '#root, #app, [id^="root-"], [id^="app-"]'
    );
    roots.forEach(el => el.setAttribute('data-page-agent-not-interactive', 'true'));
}"""

_PATCH_ANTD_JS = """() => {
    const selectors = [
        '.ant-select-selector',
        '.ant-select-dropdown',
        '.ant-select-item-option',
        '.ant-picker-panel',
        '.ant-picker-dropdown',
        '.ant-cascader-menus',
        '.ant-tree-select-dropdown'
    ];
    selectors.forEach(sel => {
        document.querySelectorAll(sel).forEach(el => {
            if (el instanceof HTMLElement) {
                el.setAttribute('data-page-agent-not-interactive', 'true');
            }
        });
    });
}"""

_CLICK_BY_XPATH_JS = """(xpath) => {
    const byXpath = (xp) => {
        if (!xp) return null;
        try {
            return document.evaluate(
                xp,
                document,
                null,
                XPathResult.FIRST_ORDERED_NODE_TYPE,
                null
            ).singleNodeValue;
        } catch (e) {
            return null;
        }
    };
    const el = byXpath(xpath);
    if (!(el instanceof HTMLElement)) {
        return JSON.stringify({success: false, reason: 'not-found'});
    }
    try {
        el.scrollIntoView({block: 'center', inline: 'nearest'});
    } catch (e) {}
    try {
        el.click();
        return JSON.stringify({success: true, tag: (el.tagName || '').toLowerCase()});
    } catch (e) {
        return JSON.stringify({success: false, reason: String(e)});
    }
}"""

_EXTRACT_MENU_JS = """() => {
    // Walk all menu containers on the page and extract hierarchical structure.
    // Supports Ant Design, Element UI, role-based menus, and native nav elements.
    const normalizeText = (s) => (s || '').replace(/\\s+/g, ' ').trim();

    const containerSelectors = [
        '[role="menubar"]', '[role="menu"]',
        '.ant-menu-root', '.ant-menu',
        '.el-menu', '.el-menu--horizontal', '.el-menu--vertical',
        'nav[aria-label]', 'nav.navbar', 'nav.sidebar',
        '.sidebar-menu', '.main-menu', '.nav-menu',
    ];

    const itemSelectors = [
        '[role="menuitem"]',
        '.ant-menu-item', '.ant-menu-submenu', '.ant-menu-submenu-title',
        '.el-menu-item', '.el-submenu', '.el-submenu__title',
        'a', 'button', '[role="link"]', '[role="button"]',
    ];

    const buildNode = (el, level) => {
        const tag = (el.tagName || '').toLowerCase();
        const text = normalizeText(el.textContent || '');
        if (!text) return null;

        // Extract href from anchor or descendant anchor
        let href = '';
        if (tag === 'a') {
            href = (el).href || (el).getAttribute('href') || '';
        } else {
            const anchor = el.querySelector('a');
            if (anchor) href = anchor.href || anchor.getAttribute('href') || '';
        }

        // Clean href: keep only path (+ hash, - origin)
        if (href && href.startsWith('http')) {
            try { href = new URL(href).pathname + new URL(href).search + new URL(href).hash; } catch(e) {}
        } else if (href && !href.startsWith('/') && !href.startsWith('#')) {
            href = '/' + href;
        }

        return {
            text: text.length > 80 ? text.slice(0, 80) + '...' : text,
            href: href,
            level: level,
            tag: tag,
            children: [],
        };
    };

    const extractChildren = (container, level) => {
        const results = [];
        // Direct children of the container that are menu items
        const candidates = Array.from(container.children);
        for (const child of candidates) {
            // Skip separator elements
            if (child.getAttribute('role') === 'separator') continue;
            if (child.classList.contains('ant-menu-item-divider')) continue;
            if (child.classList.contains('el-menu-item-group__title')) continue;

            const node = buildNode(child, level);
            if (!node) continue;

            // Check for submenu — container has nested items
            let subContainer = null;
            if (child.classList.contains('ant-menu-submenu')) {
                subContainer = child.querySelector(':scope > .ant-menu');
            } else if (child.classList.contains('el-submenu')) {
                subContainer = child.querySelector(':scope > .el-menu');
            } else if (child.getAttribute('aria-haspopup') === 'true' || child.getAttribute('aria-expanded') !== null) {
                subContainer = child.querySelector('[role="menu"]');
            }

            if (subContainer) {
                node.children = extractChildren(subContainer, level + 1);
            }

            results.push(node);
        }
        return results;
    };

    // Find best container
    let mainContainer = null;
    for (const sel of containerSelectors) {
        const el = document.querySelector(sel);
        if (el) { mainContainer = el; break; }
    }
    // Fallback: first <nav> element
    if (!mainContainer) {
        const nav = document.querySelector('nav');
        if (nav) mainContainer = nav;
    }

    const items = mainContainer ? extractChildren(mainContainer, 1) : [];
    // If main container is deep in DOM (e.g. Ant Design's .ant-menu-root inside a wrapper),
    // and we got few items, try finding sub-containers
    if (items.length === 0 && mainContainer) {
        const innerMenus = mainContainer.querySelectorAll('.ant-menu, .el-menu, [role="menu"]');
        for (const m of innerMenus) {
            if (m === mainContainer) continue;
            const sub = extractChildren(m, 1);
            if (sub.length > items.length) { items.length = 0; items.push(...sub); }
        }
    }

    return JSON.stringify({
        items: items,
        found_container: !!mainContainer,
        hint: items.length === 0 ? (
            'No menu items found in current DOM. '
            + (mainContainer ? 'Menu container exists but no items detected.' : 'No menu container element found.')
            + ' The menu may require opening first (try a keyboard shortcut like Alt+Z or clicking a menu trigger).'
            + ' After opening, call extract_menu again.'
        ) : null
    });
}"""

_TOP_LAYER_INFO_JS = """(el) => {
    if (!(el instanceof HTMLElement)) {
        return JSON.stringify({is_visible: false, is_top: false, reason: 'not-html'});
    }
    const rect = el.getBoundingClientRect();
    const isVisible = !!(
        rect &&
        rect.width > 0 &&
        rect.height > 0 &&
        rect.bottom >= 0 &&
        rect.top <= window.innerHeight &&
        rect.right >= 0 &&
        rect.left <= window.innerWidth
    );
    if (!isVisible) {
        return JSON.stringify({
            is_visible: false,
            is_top: false,
            reason: 'out-of-viewport',
            bbox: {left: rect.left, top: rect.top, width: rect.width, height: rect.height}
        });
    }
    const margin = 5;
    const checkPoints = [
        {x: rect.left + rect.width / 2, y: rect.top + rect.height / 2},
        {x: rect.left + margin, y: rect.top + margin},
        {x: rect.right - margin, y: rect.bottom - margin},
    ];
    let hitSource = 'document';
    const rootNode = el.getRootNode && el.getRootNode();

    const isInAncestorChain = (topEl, targetEl, stopAt) => {
        let cur = topEl;
        while (cur && cur !== stopAt) {
            if (cur === targetEl) return true;
            cur = cur.parentElement;
        }
        return false;
    };

    const checkPoint = ({x, y}) => {
        try {
            if (rootNode instanceof ShadowRoot && typeof rootNode.elementFromPoint === 'function') {
                const shadowTop = rootNode.elementFromPoint(x, y);
                if (shadowTop && isInAncestorChain(shadowTop, el, rootNode)) {
                    hitSource = 'shadow';
                    return true;
                }
            }
        } catch (e) {}
        try {
            const topEl = document.elementFromPoint(x, y);
            if (!topEl) return false;
            return isInAncestorChain(topEl, el, document.documentElement);
        } catch (e) {
            return true;
        }
    };

    const isTop = checkPoints.some(checkPoint);
    return JSON.stringify({
        is_visible: true,
        is_top: isTop,
        hit_source: hitSource,
        bbox: {left: rect.left, top: rect.top, width: rect.width, height: rect.height}
    });
}"""

_PATCH_REQUIRED_FIELDS_JS = """() => {
    // Mark required form fields so the LLM can see which inputs must be filled.
    // Broad detection: scans all labels for required indicators (`::before *`,
    // aria-required, class markers, etc.), then finds the nearest associated
    // input via any structural relationship — no hardcoded DOM patterns.
    var processed = new Set();
    var i, el, label, name, ph, req, bc, ac, txt;

    var cleanText = function(s) {
        return (s || '').replace(/[*:\\uFF1A\\s*]+$/g, '').trim();
    };

    var isMarkedRequired = function(lbl) {
        if (!lbl) return false;
        if (lbl.classList.contains('ant-form-item-required')) return true;
        var fi = lbl.closest('.ant-form-item, .el-form-item, [class*="form-item" i]');
        if (fi && (fi.classList.contains('ant-form-item-required') || fi.classList.contains('is-required'))) return true;
        try { bc = window.getComputedStyle(lbl, '::before').content; }
        catch(e) { bc = 'none'; }
        if (bc && bc !== 'none' && bc !== 'normal' && bc.indexOf('*') !== -1) return true;
        try { ac = window.getComputedStyle(lbl, '::after').content; }
        catch(e) { ac = 'none'; }
        if (ac && ac !== 'none' && ac !== 'normal' && ac.indexOf('*') !== -1) return true;
        txt = (lbl.textContent || '').trim();
        if (txt && (txt.charAt(txt.length - 1) === '*' || /[\\u2605\\u2731\\u2732\\u2734\\u2736\\u2737\\u2738\\u2739]/.test(txt))) return true;
        return false;
    };

    // Walk up the DOM tree to find the nearest input within the same form/field container.
    var findAssociatedInput = function(start) {
        // Strategy 1: label[for] -> input[id]
        var labelFor = start.closest('label');
        if (labelFor) {
            var fid = labelFor.getAttribute('for');
            if (fid) {
                var target = document.getElementById(fid);
                if (target) return target;
            }
        }
        // Strategy 2: shared form-item container
        var container = start.closest('.ant-form-item, .el-form-item, .form-group, .form-item, .field, [class*="form-item" i], form, .ant-form, .el-form, fieldset');
        if (container) {
            var inp = container.querySelector('input, select, textarea');
            if (inp) return inp;
        }
        // Strategy 3: sibling/parent walk — look for input nearby
        var el = start;
        for (var depth = 0; depth < 4 && el; depth++) {
            var parent = el.parentElement;
            if (!parent) break;
            var inp = parent.querySelector('input, select, textarea');
            if (inp) return inp;
            el = parent;
        }
        return null;
    };

    // --- Phase 1: Scan ALL <label> elements for required markers ---
    var allLabels = document.querySelectorAll('label, [class*="label" i], .ant-form-item-label, .el-form-item__label');
    for (i = 0; i < allLabels.length; i++) {
        label = allLabels[i];
        if (!isMarkedRequired(label)) continue;

        var input = findAssociatedInput(label);
        if (!input || processed.has(input) || input.offsetParent === null) continue;

        input.setAttribute('required', '');
        input.setAttribute('aria-required', 'true');

        name = cleanText(label.textContent || '');
        if (name && !input.getAttribute('name')) {
            input.setAttribute('name', name);
        }
        ph = input.getAttribute('placeholder');
        if (name && (!ph || ph === '')) {
            input.setAttribute('placeholder', name.slice(0, 20) + ' *\\u5fc5\\u586b');
        }
        processed.add(input);
    }

    // --- Phase 2: Scan inputs for native/class-based required markers ---
    var fallback = document.querySelectorAll(
        'input[required], textarea[required], select[required], '
        + '[aria-required="true"], '
        + '.ant-form-item-required input, .ant-form-item-required textarea, '
        + '.ant-form-item-required select, '
        + '.el-form-item.is-required input, .el-form-item.is-required textarea, '
        + '.el-form-item.is-required select, '
        + '.is-required input, .is-required textarea, .is-required select'
    );
    for (i = 0; i < fallback.length; i++) {
        el = fallback[i];
        if (processed.has(el) || el.offsetParent === null) continue;

        if (!el.getAttribute('placeholder') || el.getAttribute('placeholder') === '') {
            var hint = '';
            var container = el.closest('.ant-form-item, .el-form-item, .form-group, .form-item, .field, [class*="form-item" i]');
            if (container) {
                var lbl = container.querySelector('label, .ant-form-item-label, .el-form-item__label, [class*="label" i]');
                if (lbl) hint = cleanText(lbl.textContent || '');
            }
            if (!hint) {
                var prev = el.previousElementSibling;
                if (prev) hint = cleanText(prev.textContent || '');
            }
            if (hint) el.setAttribute('placeholder', hint.slice(0, 20) + ' *\\u5fc5\\u586b');
        }
        el.setAttribute('required', '');
        el.setAttribute('aria-required', 'true');
        processed.add(el);
    }

    var results = [];
    processed.forEach(function(el) {
        results.push({
            tag: (el.tagName || '').toLowerCase(),
            id: el.id || '',
            name: el.getAttribute('name') || '',
            required: el.hasAttribute('required'),
            placeholder: (el.getAttribute('placeholder') || '').slice(0, 30)
        });
    });
    return JSON.stringify(results);
}"""
