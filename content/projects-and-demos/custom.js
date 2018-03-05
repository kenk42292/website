/* global $, document */
/* Filter out  */
$(document).ready(function() {
    'use strict';
    $('.proj-demo-search-input').on('input', function () {
        var filter = $(this).val();
        var entries = $('.card');
        $(entries).show();
        var i;
        var entry;
        var inTitle, inDesc;
        for (i = 0; i < entries.length; i++) {
            entry = entries[i];
            inTitle = $(entry).find(".entry-title:containsi(" + filter + ")");
            inDesc = $(entry).find(".entry-desc:containsi(" + filter + ")");
            if (!inTitle.length && !inDesc.length) {
                $(entry).css('display', 'none');
            }
        }
    });
});