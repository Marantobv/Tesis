const fade = (direction,speed) =>{
    return{
        init: {
            opacity: 0,
            y: direction === 'up' ? 100 : direction === 'down' ? -100 : 0,
            x: direction === 'right' ? -100 : direction === 'left' ? 100 : 0
        },
        view: {
            opacity: 1,
            x:0,
            y:0,
            transition:{
                duration: speed,
                delay: 0.3
            }
        }
    }
}

export {fade}