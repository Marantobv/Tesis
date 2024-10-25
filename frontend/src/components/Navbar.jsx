import React from 'react'
import { HiOutlineBriefcase, HiOutlineHome, HiOutlineUser, HiOutlineClipboardCheck } from 'react-icons/hi'
import { AiOutlineStock } from "react-icons/ai";
import { FaRegNewspaper, FaRegCalendarCheck  } from "react-icons/fa";
import {BiMessageSquareDetail } from 'react-icons/bi'
import {Link} from 'react-scroll'

function Navbar() {

  return (
    <nav className='fixed bottom-2 lg:bottom-8 w-full overflow-hidden z-30 p-4 lg:p-0'>
      <div className='container mx-auto'>
        <div className='flex items-center justify-around w-full bg-black/10 h-[65px] backdrop-blur-2xl max-w-[460px] mx-auto rounded-full border-morado border'>
          <Link activeClass='is-active' className='cursor-pointer text-morado' to='home' spy={true} smooth={true} offset={-110} duration={500}>
            <HiOutlineHome size={24}></HiOutlineHome>
          </Link>
          <Link activeClass='is-active' className='cursor-pointer text-morado' to='about' spy={true} smooth={true} offset={50} duration={500}>
            <HiOutlineUser size={24}></HiOutlineUser>
          </Link>
          <Link activeClass='is-active' className='cursor-pointer text-morado' to='charts' spy={true} smooth={true} offset={50} duration={500}>
            <AiOutlineStock size={24}></AiOutlineStock>
          </Link>
          <Link activeClass='is-active' className='cursor-pointer text-morado' to='projects' spy={true} smooth={true} offset={50} duration={500}>
            <FaRegNewspaper  size={24}></FaRegNewspaper>
          </Link>
          <Link activeClass='is-active' className='cursor-pointer text-morado' to='news-and-sentiment' spy={true} smooth={true} offset={50} duration={500}>
            <FaRegCalendarCheck size={24}></FaRegCalendarCheck>
          </Link>
        </div>
      </div>
    </nav>
  )
}

export default Navbar