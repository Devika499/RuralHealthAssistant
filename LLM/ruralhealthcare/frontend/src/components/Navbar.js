import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Box,
  Menu,
  MenuItem,
  Avatar,
  Badge,
} from '@mui/material';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import LanguageIcon from '@mui/icons-material/Language';
import NotificationsIcon from '@mui/icons-material/Notifications';
import { useAuth } from '../hooks/useAuth';
import { useLanguage } from '../hooks/useLanguage';
import { useChat } from '../hooks/useChat';
import i18n from '../i18n';

const Navbar = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { language, setLanguage } = useLanguage();
  const { messages, clearMessages } = useChat();

  // Menu states
  const [profileAnchorEl, setProfileAnchorEl] = useState(null);
  const [langAnchorEl, setLangAnchorEl] = useState(null);
  const [notifAnchorEl, setNotifAnchorEl] = useState(null);
  const [activePage, setActivePage] = useState('chat');

  // Notification reminders
  const [reminders, setReminders] = useState([]);
  const [loadingReminders, setLoadingReminders] = useState(false);

  // Current time
  const [currentTime, setCurrentTime] = useState(() => {
    const now = new Date();
    return now.toTimeString().slice(0, 5); // "HH:mm"
  });

  // Update currentTime every minute
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      setCurrentTime(now.toTimeString().slice(0, 5));
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  // Fetch reminders for the current user
  useEffect(() => {
    const fetchReminders = async () => {
      if (!user) return;
      setLoadingReminders(true);
      try {
        const token = localStorage.getItem('token');
        const res = await fetch('/medications/', {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const data = await res.json();
          setReminders(data);
        }
      } catch (err) {
        // ignore
      }
      setLoadingReminders(false);
    };
    fetchReminders();
  }, [user]);

  // Filter reminders for today
  const today = new Date().toISOString().slice(0, 10);
  const remindersDueToday = reminders.filter(
    (rem) =>
      rem.start_date.slice(0, 10) <= today &&
      (!rem.end_date || rem.end_date.slice(0, 10) >= today)
  );

  // Only show reminders due now (at this minute)
  const remindersDueNow = reminders.filter(rem => rem.time === currentTime);

  // Handlers
  const handleProfileMenu = (event) => setProfileAnchorEl(event.currentTarget);
  const handleProfileClose = () => setProfileAnchorEl(null);
  const handleLangMenu = (event) => setLangAnchorEl(event.currentTarget);
  const handleLangClose = () => setLangAnchorEl(null);
  const handleNotifClick = (event) => setNotifAnchorEl(event.currentTarget);
  const handleNotifClose = () => setNotifAnchorEl(null);

  const languages = [
    { value: 'en', label: 'English' },
    { value: 'hi', label: 'Hindi' },
    { value: 'te', label: 'Telugu' },
    { value: 'ta', label: 'Tamil' },
    { value: 'kn', label: 'Kannada' },
    { value: 'bn', label: 'Bengali' },
    { value: 'mr', label: 'Marathi' },
    { value: 'pa', label: 'Punjabi' },
    { value: 'gu', label: 'Gujarati' },
    { value: 'ml', label: 'Malayalam' },
    { value: 'or', label: 'Odia' },
  ];

  return (
    <>
      <Box
        sx={{
          width: '100%',
          bgcolor: '#1976d2',
          color: 'white',
          py: 2,
          px: 4,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          boxShadow: '0 2px 8px 0 rgba(31, 38, 135, 0.10)',
        }}
      >
        <Typography variant="h5" sx={{ fontWeight: 700, letterSpacing: 1 }}>
          Rural HealthCare Assistant
        </Typography>
        {user && (
          <Box>
            {/* Notifications icon */}
            <IconButton color="inherit" onClick={handleNotifClick}>
              <Badge badgeContent={remindersDueNow.length} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
            <Menu
              anchorEl={notifAnchorEl}
              open={Boolean(notifAnchorEl)}
              onClose={handleNotifClose}
            >
              {loadingReminders ? (
                <MenuItem>Loading...</MenuItem>
              ) : remindersDueNow.length === 0 ? (
                <MenuItem>No reminders due now</MenuItem>
              ) : (
                remindersDueNow.map((rem, idx) => (
                  <MenuItem key={idx}>
                    {rem.medication_name} at {rem.time}
                  </MenuItem>
                ))
              )}
            </Menu>
            {/* Language menu button */}
            <IconButton color="inherit" onClick={handleLangMenu}>
              <LanguageIcon />
            </IconButton>
            <Menu
              anchorEl={langAnchorEl}
              open={Boolean(langAnchorEl)}
              onClose={handleLangClose}
              sx={{ zIndex: 1302 }}
            >
              {languages.map((lang) => (
                <MenuItem
                  key={lang.value}
                  onClick={() => {
                    setLanguage(lang.value);
                    i18n.changeLanguage(lang.value);
                    handleLangClose();
                  }}
                  selected={language === lang.value}
                >
                  {lang.label}
                </MenuItem>
              ))}
            </Menu>
            {/* Profile menu button */}
            <IconButton color="inherit" onClick={handleProfileMenu}>
              <Avatar alt={user.name} sx={{ bgcolor: '#1976d2' }}>
                {user.name[0]}
              </Avatar>
            </IconButton>
            <Menu
              anchorEl={profileAnchorEl}
              open={Boolean(profileAnchorEl)}
              onClose={handleProfileClose}
              sx={{ zIndex: 1301, mt: 1 }}
            >
              <MenuItem
                onClick={() => {
                  handleProfileClose();
                  navigate('/profile');
                }}
              >
                Profile
              </MenuItem>
              <MenuItem
                onClick={() => {
                  handleProfileClose();
                  navigate('/medical-history');
                }}
              >
                Medical History
              </MenuItem>
              <MenuItem
                onClick={async () => {
                  await logout(messages, clearMessages);
                  handleProfileClose();
                }}
              >
                Logout
              </MenuItem>
            </Menu>
          </Box>
        )}
        {!user && (
          <Box>
            <Button color="inherit" onClick={() => navigate('/login')}>
              Login
            </Button>
            <Button color="inherit" onClick={() => navigate('/register')}>
              Register
            </Button>
          </Box>
        )}
      </Box>
      {/* Buttons below navbar for Chatbot, Medical Term Simplifier, Symptom Checker, and Add Medication */}
      {user && (
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 3, mb: 3 }}>
          <Button
            variant={activePage === 'chat' ? 'contained' : 'outlined'}
            color="primary"
            sx={{ mx: 1, minWidth: 160, fontWeight: 600, letterSpacing: 1 }}
            onClick={() => {
              setActivePage('chat');
              navigate('/chat');
            }}
          >
            CHATBOT
          </Button>
          <Button
            variant={activePage === 'simplify' ? 'contained' : 'outlined'}
            color="info"
            sx={{ mx: 1, minWidth: 220, fontWeight: 600, letterSpacing: 1 }}
            onClick={() => {
              setActivePage('simplify');
              navigate('/simplify-term');
            }}
          >
            MEDICAL TERM SIMPLIFIER
          </Button>
          <Button
            variant={activePage === 'symptom' ? 'contained' : 'outlined'}
            color="success"
            sx={{ mx: 1, minWidth: 200, fontWeight: 600, letterSpacing: 1 }}
            onClick={() => {
              setActivePage('symptom');
              navigate('/symptom-checker');
            }}
          >
            SYMPTOM CHECKER
          </Button>
          <Button
            variant={activePage === 'medication' ? 'contained' : 'outlined'}
            color="secondary"
            sx={{ mx: 1, minWidth: 180, fontWeight: 600, letterSpacing: 1 }}
            onClick={() => {
              setActivePage('medication');
              navigate('/medications');
            }}
          >
            ADD MEDICATION
          </Button>
          <Button
            variant={activePage === 'diet' ? 'contained' : 'outlined'}
            color="warning"
            sx={{ mx: 1, minWidth: 220, fontWeight: 600, letterSpacing: 1 }}
            onClick={() => {
              setActivePage('diet');
              navigate('/diet-recommendation');
            }}
          >
            DIET RECOMMENDATION
          </Button>
        </Box>
      )}
    </>
  );
};

export default Navbar;