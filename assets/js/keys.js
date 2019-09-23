$(document).ready(() => {
    $('pre').each(function(tag) {
        $(this).load(`/assets/data/keys/${$(this).data('name')}`);
    });
});
