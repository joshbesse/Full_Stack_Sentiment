import React from 'react';
import { Link } from 'react-router-dom';

function Navbar() {
    return (
        <header>
            <nav className='navbar'>
                <ul>
                    <li><Link to='/home'>Home</Link></li>
                    <li><Link to='/description'>Description</Link></li>
                    <li><Link to='/history'>History</Link></li>
                </ul>
            </nav>
        </header>
    )
}

export default Navbar