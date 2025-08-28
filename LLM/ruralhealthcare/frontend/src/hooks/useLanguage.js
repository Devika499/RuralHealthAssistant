import { useState, useEffect } from 'react';
import { useAuth } from './useAuth';

export const useLanguage = () => {
  const { user, updateUser } = useAuth();
  const [language, setLanguage] = useState('hi'); // Default to Hindi

  useEffect(() => {
    if (user) {
      setLanguage(user.preferredLanguage || 'hi');
    }
  }, [user]);

  const handleLanguageChange = async (newLanguage) => {
    if (user) {
      await updateUser({ preferredLanguage: newLanguage });
      setLanguage(newLanguage);
    }
  };

  return { language, setLanguage: handleLanguageChange };
};
