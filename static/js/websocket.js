
$(document).ready(function(){


 var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
    socket.on('my response', function(msg) {
        $('#log').append('<p>Received: ' + msg.data + '</p>');
    });
    $('form#emit').submit(function(event) {
        socket.emit('my event', {data: $('#emit_data').val()});
        return false;
    });
    $('form#broadcast').submit(function(event) {
        socket.emit('my broadcast event', {data: $('#broadcast_data').val()});
        return false;
    });




//        var ws = new WebSocket('ws://' + document.domain + ':' + location.port + "/message");
//       ws.onopen = function() {
//            ws.send("hello")
//           alert("Open");
//       };
//       ws.onclose = function() {
//           alert("Close");
//       };
//       ws.onmessage = function(evt) {
//           alert(evt.data);
//       };








});



// var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
//    socket.on('my response', function(msg) {
//        $('#log').append('<p>Received: ' + msg.data + '</p>');
//    });
//    $('form#emit').submit(function(event) {
//        socket.emit('my event', {data: $('#emit_data').val()});
//        return false;
//    });
//    $('form#broadcast').submit(function(event) {
//        socket.emit('my broadcast event', {data: $('#broadcast_data').val()});
//        return false;
//    });