// Hack to change the max-width of the divs containing the messages.
// Its so complicated since `chainlit` keeps undoing any basic changes.
window.onload = function() {
    const updateDivStyles = (div) => {
        if (div.style.maxWidth === 'min(48rem, 100vw)') {
            div.style.maxWidth = '60vw';
        }
    };

    const observeDiv = (div) => {
        updateDivStyles(div);

        const observer = new MutationObserver(() => {
            updateDivStyles(div);
        });

        observer.observe(div, {
            attributes: true,
            attributeFilter: ['style']
        });
    };

    // Initial update for existing divs
    const divs = document.querySelectorAll('div.flex.flex-col');
    divs.forEach(observeDiv);

    // Use MutationObserver to watch for new divs being added to the DOM
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === 1 && node.matches('div.flex.flex-col')) {
                    observeDiv(node);
                }
            });

            // Also check for changes in existing nodes
            mutation.target.querySelectorAll('div.flex.flex-col').forEach(observeDiv);
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
};
