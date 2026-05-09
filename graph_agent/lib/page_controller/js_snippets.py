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
